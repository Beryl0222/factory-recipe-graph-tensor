"""分段 TopK 解码算子的行为测试。"""
import unittest
from pathlib import Path

from segbeam import (
    Draft,
    InvalidatedSession,
    PoolStatus,
    PrematureFinalization,
    SegmentedTopKSession,
    ShardInconsistency,
    WorkOrderGrammar,
)
from segbeam.adapter import replay

DATA = Path(__file__).resolve().parents[1] / "data"


def sample_grammar() -> WorkOrderGrammar:
    """与 data/capture.json 一致的工单语法。"""
    return WorkOrderGrammar(
        transitions=[
            ("root", 1, "prepA", "load"),
            ("root", 2, "prepB", "load"),
            ("root", 6, "inspect", "inspect"),
            ("prepA", 3, "doneA", "seal"),
            ("prepB", 3, "doneB", "seal"),
            ("inspect", 4, "deep", "probe"),
            ("deep", 3, "doneA", "seal"),
            ("doneA", 8, "doneA", "report"),
            ("doneB", 8, "doneB", "report"),
        ],
        accepting=["doneA", "doneB"],
        equivalence={
            "root": "r",
            "prepA": "p",
            "prepB": "p",
            "inspect": "i",
            "doneA": "d",
            "doneB": "d",
            "deep": "x",
        },
    )


def beam(identity, state, score=0, words=(), ops=()):
    return Draft(identity=identity, state=state, words=words, ops=ops, score=score)


class ReplayTests(unittest.TestCase):
    """按采集记录完整回放，核对每轮定案结果与最终提交。"""

    def setUp(self):
        self.report = replay(DATA / "capture.json")

    def test_round_zero_synonyms_do_not_crowd_out(self):
        feeds = self.report.rounds[0].feeds
        # 只到 early 分片：late 上界 31 与现任同分，纯分数界不能排除未决路径
        early = feeds[0].status
        self.assertFalse(early.continuation.finalized)
        self.assertEqual(early.continuation.pending_shards, ("late",))
        self.assertEqual([c.words for c in early.continuation.entries], [(2,)])
        self.assertEqual(early.done.status, PoolStatus.PENDING_SHARDS)
        # late 到齐：同义写法 1/2 竞争同一席位，词序列字典序决定；inspect 草稿保留
        late = feeds[1].status
        self.assertTrue(late.continuation.finalized)
        self.assertEqual([c.words for c in late.continuation.entries], [(1,), (6,)])
        self.assertEqual([c.score for c in late.continuation.entries], [31, 29])
        # 词 4 在 root 不能消费，45 分不参与竞争
        self.assertNotIn((4,), [c.words for c in late.continuation.entries])
        self.assertEqual(late.done.status, PoolStatus.EMPTY_NO_GRAMMAR_CANDIDATES)

    def test_round_one_early_finalization_by_bounds(self):
        round_one = self.report.rounds[1]
        main = round_one.feeds[0].status
        # extra 分片未到，但上界 7 证明它改变不了席位：提前定案
        self.assertTrue(main.continuation.finalized)
        self.assertEqual([c.words for c in main.continuation.entries], [(1, 3), (6, 4)])
        # 可提前终止的命令既入完成池，也继续占一个续写席位
        self.assertEqual([e.identity for e in main.done.entries], [("load", "seal")])
        outcome = round_one.outcome
        self.assertEqual([b.identity for b in outcome.next_beams], [(1, 3), (6, 4)])
        self.assertEqual(outcome.done_pool[0].words, (1, 3))

    def test_round_two_carried_done_keeps_competing(self):
        outcome = self.report.rounds[2].outcome
        # 以前各轮已定案的完成项继续竞争，(load, seal) 被两条更优路径挤出
        self.assertEqual(
            [e.identity for e in outcome.done_pool],
            [("inspect", "probe", "seal"), ("load", "seal", "report")],
        )
        self.assertEqual([e.score for e in outcome.done_pool], [48, 41])

    def test_submission_with_provenance(self):
        sub = self.report.submission
        self.assertIsNotNone(sub)
        self.assertEqual(sub.command, ("inspect", "probe", "seal"))
        self.assertEqual(sub.words, (6, 4, 3))
        self.assertEqual(sub.score, 48)
        self.assertEqual(sub.sources, ("b0", (6,), (6, 4)))


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.grammar = sample_grammar()

    def start_round_zero(self, session):
        session.begin_round(
            [beam("b0", "root")],
            remaining=3,
            beam_capacity=2,
            done_capacity=2,
            unseen={"late": {"words": [1, 6], "upper": 31}},
        )

    def test_premature_finalization_rejected(self):
        session = SegmentedTopKSession(self.grammar)
        self.start_round_zero(session)
        session.feed("early", [("b0", 2, 31), ("b0", 4, 45)])
        with self.assertRaises(PrematureFinalization):
            session.finalize_round()
        session.feed("late", [("b0", 1, 31), ("b0", 6, 29)])
        outcome = session.finalize_round()
        self.assertEqual([b.identity for b in outcome.next_beams], [(1,), (6,)])

    def test_shard_retransmission_is_idempotent(self):
        session = SegmentedTopKSession(self.grammar)
        self.start_round_zero(session)
        session.feed("early", [("b0", 2, 31), ("b0", 4, 45)])
        before = session.status()
        session.feed("early", [("b0", 2, 31), ("b0", 4, 45)])  # 重发：空操作
        after = session.status()
        self.assertEqual(
            [c.words for c in before.continuation.entries],
            [c.words for c in after.continuation.entries],
        )
        with self.assertRaises(ShardInconsistency):
            session.feed("early", [("b0", 2, 30)])  # 重发内容不一致
        self.assertFalse(session.invalidated)  # 轮次未定案，可放弃重来
        session.abort_round()

    def test_watched_shard_violation_invalidates_session(self):
        session = SegmentedTopKSession(self.grammar)
        self.start_round_zero(session)
        session.feed("early", [("b0", 2, 31), ("b0", 4, 45)])
        session.feed("late", [("b0", 1, 31), ("b0", 6, 29)])
        outcome0 = session.finalize_round()
        session.begin_round(
            list(outcome0.next_beams),
            remaining=2,
            beam_capacity=2,
            done_capacity=2,
            unseen={"extra": {"words": [3, 4, 8], "upper": 7}},
        )
        session.feed("main", [((1,), 3, 9), ((6,), 4, 7)])
        outcome1 = session.finalize_round()  # 凭上界提前定案，extra 转入监视
        self.assertEqual(outcome1.watched_shards, ("extra",))
        with self.assertRaises(ShardInconsistency):
            session.feed("extra", [((1,), 8, 99)])  # 超界：提前定案失效
        self.assertTrue(session.invalidated)
        with self.assertRaises(InvalidatedSession):
            session.begin_round([], remaining=1, beam_capacity=1, done_capacity=1)

    def test_watched_shard_within_bounds_keeps_outcome(self):
        session = SegmentedTopKSession(self.grammar)
        self.start_round_zero(session)
        session.feed("early", [("b0", 2, 31), ("b0", 4, 45)])
        session.feed("late", [("b0", 1, 31), ("b0", 6, 29)])
        outcome0 = session.finalize_round()
        session.begin_round(
            list(outcome0.next_beams),
            remaining=2,
            beam_capacity=2,
            done_capacity=2,
            unseen={"extra": {"words": [3, 4, 8], "upper": 7}},
        )
        session.feed("main", [((1,), 3, 9), ((6,), 4, 7)])
        session.finalize_round()
        session.feed("extra", [((1,), 8, 7), ((6,), 3, 6)])  # 迟到但在界内
        self.assertFalse(session.invalidated)
        sub = session.submit()
        self.assertEqual(sub.command, ("load", "seal"))
        # 已核验分片重发内容一致是空操作；内容不一致则报资料不一致并失效
        session.feed("extra", [((1,), 8, 7), ((6,), 3, 6)])
        self.assertFalse(session.invalidated)
        with self.assertRaises(ShardInconsistency):
            session.feed("extra", [((1,), 8, 7), ((6,), 3, 5)])
        self.assertTrue(session.invalidated)

    def test_tie_bound_direction_decides_finalization(self):
        # 未决分片上界与现任同分：词序列更大则不改写（可定案），更小则不能排除
        session = SegmentedTopKSession(self.grammar)
        session.begin_round(
            [beam("b0", "root")],
            remaining=3,
            beam_capacity=1,
            done_capacity=1,
            unseen={"late": {"words": [2], "upper": 31}},
        )
        session.feed("early", [("b0", 1, 31)])
        status = session.status()
        self.assertTrue(status.continuation.finalized)  # (2,) 字典序在 (1,) 之后
        outcome = session.finalize_round()
        self.assertEqual([b.identity for b in outcome.next_beams], [(1,)])
        session.feed("late", [("b0", 2, 31)])  # 迟到分片与证明一致
        self.assertFalse(session.invalidated)

    def test_submit_by_identity_and_unknown(self):
        session = SegmentedTopKSession(self.grammar)
        session.begin_round(
            [beam("b0", "root")], remaining=3, beam_capacity=2, done_capacity=2
        )
        session.feed("all", [("b0", 1, 31), ("b0", 2, 31), ("b0", 6, 29)])
        outcome0 = session.finalize_round()
        session.begin_round(
            list(outcome0.next_beams), remaining=2, beam_capacity=2, done_capacity=2
        )
        session.feed("main", [((1,), 3, 9), ((6,), 4, 7)])
        outcome1 = session.finalize_round()
        session.begin_round(
            list(outcome1.next_beams), remaining=1, beam_capacity=2, done_capacity=2
        )
        session.feed("base", [((6, 4), 3, 12), ((1, 3), 8, 1)])
        session.finalize_round()
        sub = session.submit(identity=("load", "seal", "report"))
        self.assertEqual(sub.words, (1, 3, 8))
        self.assertEqual(sub.score, 41)
        self.assertEqual(sub.sources, ("b0", (1,), (1, 3)))
        with self.assertRaises(KeyError):
            session.submit(identity=("nope",))

    def test_feed_without_round_rejected(self):
        session = SegmentedTopKSession(self.grammar)
        with self.assertRaises(RuntimeError):
            session.feed("x", [("b0", 1, 1)])

    def test_begin_round_validation(self):
        session = SegmentedTopKSession(self.grammar)
        with self.assertRaises(ValueError):
            session.begin_round([], remaining=0, beam_capacity=1, done_capacity=1)
        session.begin_round([beam("b0", "root")], remaining=1, beam_capacity=1, done_capacity=1)
        with self.assertRaises(RuntimeError):
            session.begin_round([], remaining=1, beam_capacity=1, done_capacity=1)


class SeatIdentityTests(unittest.TestCase):
    """席位身份：规范操作序列 + 状态等价类。"""

    def grammar(self):
        return WorkOrderGrammar(
            transitions=[
                ("root", 1, "sA", "load"),
                ("root", 2, "sB", "load"),
                ("root", 3, "sC", "load"),
                ("sA", 9, "sA", ""),
                ("sB", 9, "sB", ""),
                ("sC", 9, "sC", ""),
            ],
            accepting=["sA", "sB", "sC"],
            equivalence={"root": "r", "sA": "cA", "sB": "cB", "sC": "cC"},
        )

    def test_same_ops_different_followup_keep_identities(self):
        # 显示字符串相同（操作都是 load），但后续合法范围不同：各自身份都保留
        session = SegmentedTopKSession(self.grammar())
        session.begin_round(
            [beam("b0", "root")], remaining=2, beam_capacity=3, done_capacity=3
        )
        session.feed("s", [("b0", 1, 10), ("b0", 2, 20), ("b0", 3, 30)])
        outcome = session.finalize_round()
        self.assertEqual([b.identity for b in outcome.next_beams], [(3,), (2,), (1,)])

    def test_capacity_selects_best_across_seats(self):
        session = SegmentedTopKSession(self.grammar())
        session.begin_round(
            [beam("b0", "root")], remaining=2, beam_capacity=2, done_capacity=3
        )
        session.feed("s", [("b0", 1, 10), ("b0", 2, 20), ("b0", 3, 30)])
        outcome = session.finalize_round()
        self.assertEqual([b.identity for b in outcome.next_beams], [(3,), (2,)])

    def test_tie_breaks_by_words_then_source(self):
        grammar = WorkOrderGrammar(
            transitions=[("root", 5, "s", "go"), ("s", 9, "s", "")],
            accepting=["s"],
            equivalence={"root": "r", "s": "c"},
        )
        session = SegmentedTopKSession(grammar)
        session.begin_round(
            [beam("b0", "root"), beam("b1", "root")],
            remaining=2,
            beam_capacity=1,
            done_capacity=1,
        )
        session.feed("s", [("b1", 5, 10), ("b0", 5, 10)])
        outcome = session.finalize_round()
        # 同分同词序列：按来源 ID 排序，b0 优先
        self.assertEqual(len(outcome.next_beams), 1)
        self.assertEqual(outcome.next_beams[0].sources[0], "b0")


class DonePoolTests(unittest.TestCase):
    def test_done_merges_by_canonical_identity(self):
        grammar = WorkOrderGrammar(
            transitions=[
                ("root", 1, "dx", "seal"),
                ("root", 2, "dy", "seal"),
                ("dx", 9, "dx", ""),
                ("dy", 9, "dy", ""),
            ],
            accepting=["dx", "dy"],
            equivalence={"root": "r", "dx": "d", "dy": "d"},
        )
        session = SegmentedTopKSession(grammar)
        session.begin_round(
            [beam("b0", "root")], remaining=2, beam_capacity=2, done_capacity=1
        )
        session.feed("s", [("b0", 1, 10), ("b0", 2, 20)])
        outcome = session.finalize_round()
        # 同一规范工单身份归并，较优实参路径提供词序列
        self.assertEqual(len(outcome.done_pool), 1)
        self.assertEqual(outcome.done_pool[0].identity, ("seal",))
        self.assertEqual(outcome.done_pool[0].words, (2,))
        self.assertEqual(outcome.done_pool[0].score, 20)

    def test_carried_done_survives_without_better_challenger(self):
        grammar = sample_grammar()
        session = SegmentedTopKSession(grammar)
        session.begin_round(
            [beam("b0", "root")], remaining=3, beam_capacity=2, done_capacity=2
        )
        session.feed("only", [("b0", 1, 31), ("b0", 6, 29)])
        outcome0 = session.finalize_round()
        session.begin_round(
            list(outcome0.next_beams), remaining=2, beam_capacity=2, done_capacity=2
        )
        session.feed("main", [((1,), 3, 9), ((6,), 4, 7)])
        outcome1 = session.finalize_round()
        self.assertEqual([e.identity for e in outcome1.done_pool], [("load", "seal")])
        session.begin_round(
            list(outcome1.next_beams), remaining=1, beam_capacity=2, done_capacity=2
        )
        # 本轮完成候选分数都低于已定案的 (load, seal)=40
        session.feed("base", [((6, 4), 3, 1), ((1, 3), 8, -2)])
        outcome2 = session.finalize_round()
        identities = [e.identity for e in outcome2.done_pool]
        self.assertIn(("load", "seal"), identities)  # 已定案完成项继续竞争并保留
        carried = outcome2.done_pool[identities.index(("load", "seal"))]
        self.assertEqual(carried.words, (1, 3))
        self.assertEqual(carried.score, 40)
        self.assertEqual(outcome2.done_pool[0].identity, ("load", "seal"))


class EmptyPoolReasonTests(unittest.TestCase):
    def test_grammar_level_empty(self):
        # 候选能消费，但落点语法上不能再接词：续写池空属语法层面
        grammar = WorkOrderGrammar(
            transitions=[("root", 1, "dz", "seal")],
            accepting=["dz"],
            equivalence={"root": "r", "dz": "d"},
        )
        session = SegmentedTopKSession(grammar)
        session.begin_round(
            [beam("b0", "root")], remaining=3, beam_capacity=1, done_capacity=1
        )
        session.feed("s", [("b0", 1, 10)])
        status = session.status()
        self.assertTrue(status.continuation.finalized)
        self.assertEqual(
            status.continuation.status, PoolStatus.EMPTY_NO_GRAMMAR_CANDIDATES
        )
        self.assertEqual(status.done.status, PoolStatus.OK)  # 完成池不受影响

    def test_budget_exhausted_empty(self):
        # 候选能消费，但剩余步内到不了合法终态
        grammar = WorkOrderGrammar(
            transitions=[
                ("root", 1, "m1", "a"),
                ("m1", 2, "m2", "b"),
                ("m2", 3, "dz", "c"),
            ],
            accepting=["dz"],
            equivalence={"root": "r", "m1": "m", "m2": "n", "dz": "d"},
        )
        session = SegmentedTopKSession(grammar)
        session.begin_round(
            [beam("b0", "root")], remaining=1, beam_capacity=1, done_capacity=1
        )
        session.feed("s", [("b0", 1, 10)])
        status = session.status()
        self.assertEqual(status.continuation.status, PoolStatus.EMPTY_BUDGET_EXHAUSTED)
        self.assertEqual(
            status.done.status, PoolStatus.EMPTY_NO_GRAMMAR_CANDIDATES
        )

    def test_pending_shards_empty(self):
        grammar = sample_grammar()
        session = SegmentedTopKSession(grammar)
        session.begin_round(
            [beam("b0", "root")],
            remaining=3,
            beam_capacity=2,
            done_capacity=2,
            unseen={"late": {"words": [1, 6], "upper": 31}},
        )
        status = session.status()
        self.assertEqual(status.continuation.status, PoolStatus.PENDING_SHARDS)
        self.assertEqual(status.done.status, PoolStatus.PENDING_SHARDS)
        explanation = session.explain()
        self.assertEqual(
            explanation["continuation_empty_reason"], PoolStatus.PENDING_SHARDS.value
        )


class ScratchBoundTests(unittest.TestCase):
    """局部缓冲不随词表大小保留全部扩展。"""

    def test_buffers_bounded_by_grammar_not_vocab(self):
        words = 4000
        transitions = [("s1", 0, "s1", "")]
        for w in range(words):
            op = "ea" if w % 2 == 0 else "ob"
            transitions.append(("root", w, "s1", op))
        grammar = WorkOrderGrammar(
            transitions=transitions,
            accepting=["s1"],
            equivalence={"root": "r", "s1": "c"},
        )
        session = SegmentedTopKSession(grammar)
        beams = [beam(f"b{i}", "root") for i in range(4)]
        session.begin_round(
            beams,
            remaining=2,
            beam_capacity=4,
            done_capacity=2,
            unseen={"big": {"words": list(range(words)), "upper": 49}},
        )
        rows = [(f"b{i}", w, w % 50) for i in range(4) for w in range(words)]
        session.feed("big", rows)  # 16000 行，对应一个完整分片
        scratch = session.scratch_report()
        self.assertEqual(scratch["seats"], 2)  # 同义写法归并到语法界定的席位
        status = session.status()
        self.assertTrue(status.continuation.finalized)
        self.assertEqual(
            sorted(c.words for c in status.continuation.entries),
            [(48,), (49,)],  # 同分候选按完整词 ID 序列字典序取最优
        )
        outcome = session.finalize_round()
        self.assertEqual(len(outcome.next_beams), 2)


class GrammarTests(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            WorkOrderGrammar(
                transitions=[("a", 1, "b", "x"), ("a", 1, "c", "y")],
                accepting=["b"],
                equivalence={"a": "A", "b": "B", "c": "C"},
            )
        with self.assertRaises(ValueError):
            WorkOrderGrammar(
                transitions=[("a", 1, "b", "x")],
                accepting=["b"],
                equivalence={"a": "A"},
            )

    def test_reachability(self):
        grammar = sample_grammar()
        self.assertEqual(grammar.min_steps_to_accept("doneA"), 0)
        self.assertEqual(grammar.min_steps_to_accept("prepA"), 1)
        self.assertEqual(grammar.min_steps_to_accept("inspect"), 2)
        self.assertEqual(grammar.min_steps_to_accept("root"), 2)
        self.assertTrue(grammar.can_hold_seat("inspect", 2))
        self.assertFalse(grammar.can_hold_seat("inspect", 1))


if __name__ == "__main__":
    unittest.main()
