"""多轮分段 TopK 解码算子。

算子面向"模型按词表分片回传分数"的在线解码场景：

- 每一轮 :meth:`SegmentedTopKSession.begin_round` 接收当前草稿（累积分、稳定身份、
  解析位置）、现场设定的剩余解码步数与完成池/续写池容量，以及尚未到达分片的
  可能词域与增量上界。
- :meth:`SegmentedTopKSession.feed` 按分片增量消费 ``(来源草稿, 候选词, 定点增量)``。
  候选只在其来源草稿能消费该词时参与竞争；同一规范操作序列且同一语法状态等价类
  的候选竞争同一席位，因此同义写法不会挤掉本可续写的草稿。
- :meth:`SegmentedTopKSession.status` 报告两个池是否已定案：全部相关分片到齐，
  或用可信分数上界与语法范围证明剩余分片不能改变席位。同分时纯分数界不能把
  未决路径提前排除。
- :meth:`SegmentedTopKSession.finalize_round` 输出已定案的续写身份（供下一轮
  继续使用）与归并后的完成池；以前各轮已定案的完成项继续参加完成池竞争。
- 分片超界或重发内容不一致时抛出 :class:`ShardInconsistency`；若该轮已提前定案，
  会话标记失效，后续草稿不能消费错误定案。

局部缓冲只按"席位身份"归约候选，规模受语法（操作词表 × 状态等价类）约束，
不随词表大小保留全部扩展。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

from .grammar import WorkOrderGrammar

Identity = Hashable
Words = Tuple[int, ...]
Ops = Tuple[str, ...]


def _normalize_identity(value) -> Identity:
    """来源身份：数组形式的完整词 ID 序列归一为元组，其余身份原样保留。"""
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def _identity_sort_key(identity: Identity):
    """来源 ID 的确定序：词序列按字典序，其它身份按字符串表示，混合时词序列在前。"""
    if isinstance(identity, tuple):
        return (0, identity)
    return (1, str(identity))


def _check_word(word) -> int:
    if not isinstance(word, int) or isinstance(word, bool):
        raise ValueError(f"词 ID 应为定点整数: {word!r}")
    return word


def _check_score(score) -> int:
    if not isinstance(score, int) or isinstance(score, bool):
        raise ValueError(f"分数应为定点整数: {score!r}")
    return score


@dataclass(frozen=True)
class Draft:
    """进入一轮解码的草稿：稳定身份、解析位置、词历史、规范操作与累积分。

    ``sources`` 是从首轮草稿到自身的身份链，用于完成后回查来源。
    """

    identity: Identity
    state: Hashable
    words: Words
    ops: Ops
    score: int
    sources: Tuple[Identity, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "identity", _normalize_identity(self.identity))
        object.__setattr__(self, "words", tuple(self.words))
        object.__setattr__(self, "ops", tuple(self.ops))
        _check_score(self.score)
        if not self.sources:
            object.__setattr__(self, "sources", (self.identity,))


@dataclass(frozen=True)
class Candidate:
    """一条候选扩展：来源草稿消费一个词后的结果。"""

    words: Words
    ops: Ops
    state: Hashable
    score: int
    source: Identity
    sources: Tuple[Identity, ...]

    def sort_key(self):
        """偏好序：高分优先；同分按完整词 ID 序列字典序；再按来源 ID。"""
        return (-self.score, self.words, _identity_sort_key(self.source))

    def better_than(self, other: "Candidate") -> bool:
        return self.sort_key() < other.sort_key()


@dataclass(frozen=True)
class DoneEntry:
    """完成池条目：按完整规范工单身份归并，由较优实参路径提供词序列。"""

    identity: Ops  # 完整规范工单身份（规范操作序列）
    words: Words
    score: int
    sources: Tuple[Identity, ...]

    def sort_key(self):
        return (-self.score, self.words, tuple(_identity_sort_key(s) for s in self.sources))


@dataclass(frozen=True)
class Submission:
    """可提交的有效完成命令及其可追查来源。"""

    identity: Ops
    command: Ops
    words: Words
    score: int
    sources: Tuple[Identity, ...]


class PoolStatus(Enum):
    """池状态：已定案且有内容 / 三类空池原因 / 仍有分片可能改写结果。"""

    OK = "ok"
    EMPTY_NO_GRAMMAR_CANDIDATES = "empty_no_grammar_candidates"
    EMPTY_BUDGET_EXHAUSTED = "empty_budget_exhausted"
    PENDING_SHARDS = "pending_shards"


@dataclass(frozen=True)
class PoolView:
    """某一池在查询时刻的视图。"""

    finalized: bool
    status: PoolStatus
    entries: Tuple
    pending_shards: Tuple[str, ...]


@dataclass(frozen=True)
class RoundStatus:
    """一轮的状态：续写池与完成池视图，以及本轮已消费/未到分片。"""

    step: int
    remaining: int
    continuation: PoolView
    done: PoolView
    delivered_shards: Tuple[str, ...]
    invalidated: bool = False


@dataclass(frozen=True)
class RoundOutcome:
    """定案输出：下一轮续写身份、归并后的完成池、以及提前定案所依赖的分片声明。"""

    step: int
    next_beams: Tuple[Draft, ...]
    done_pool: Tuple[DoneEntry, ...]
    watched_shards: Tuple[str, ...]


class ShardInconsistency(Exception):
    """分片超界或重发内容不一致：依据它做的提前决定失效。"""


class PrematureFinalization(Exception):
    """尚有分片可能改写席位时要求定案。"""


class InvalidatedSession(Exception):
    """会话已因资料不一致失效，后续草稿不能消费错误定案。"""


class _UnseenDecl:
    """未到分片的可信声明：可能词域与增量上界。"""

    __slots__ = ("words", "upper")

    def __init__(self, words: Iterable[int], upper: int) -> None:
        self.words = frozenset(_check_word(w) for w in words)
        self.upper = _check_score(upper)


class _Fingerprint:
    """分片内容的顺序无关指纹，用于识别重发与内容不一致。"""

    __slots__ = ("count", "total", "mix")

    def __init__(self) -> None:
        self.count = 0
        self.total = 0
        self.mix = 0

    @staticmethod
    def _row_hash(source: Identity, word: int, delta: int) -> int:
        digest = hashlib.sha256(repr((source, word, delta)).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")

    def add(self, source: Identity, word: int, delta: int) -> None:
        h = self._row_hash(source, word, delta)
        self.count += 1
        self.total = (self.total + h) % (1 << 64)
        self.mix ^= h

    def matches(self, other: "_Fingerprint") -> bool:
        return (
            self.count == other.count
            and self.total == other.total
            and self.mix == other.mix
        )


class _Round:
    """一轮解码的可变状态。缓冲规模受语法约束，不随词表大小增长。"""

    def __init__(
        self,
        step: int,
        beams: Sequence[Draft],
        remaining: int,
        beam_capacity: int,
        done_capacity: int,
        unseen: Mapping[str, _UnseenDecl],
    ) -> None:
        self.step = step
        self.remaining = remaining
        self.beam_capacity = beam_capacity
        self.done_capacity = done_capacity
        self.beams: Dict[Identity, Draft] = {}
        for beam in beams:
            if beam.identity in self.beams:
                raise ValueError(f"草稿身份重复: {beam.identity!r}")
            self.beams[beam.identity] = beam
        # 续写席位：(规范操作序列, 状态等价类) -> 当前最优候选
        self.seat_best: Dict[Tuple[Ops, Hashable], Candidate] = {}
        # 完成候选：完整规范工单身份 -> 当前最优实参路径
        self.done_best: Dict[Ops, Candidate] = {}
        self.pending: Dict[str, _UnseenDecl] = dict(unseen)
        self.delivered: Dict[str, _Fingerprint] = {}
        self.saw_grammar_blocked = False  # 候选能消费，但落点语法上不能再接词
        self.saw_budget_blocked = False  # 候选能消费，但剩余步内到不了合法终态

    def seat_winners(self) -> List[Candidate]:
        winners = sorted(self.seat_best.values(), key=Candidate.sort_key)
        return winners[: self.beam_capacity]

    def done_identities(self, carried: Sequence[DoneEntry]) -> Dict[Ops, DoneEntry]:
        merged: Dict[Ops, DoneEntry] = {entry.identity: entry for entry in carried}
        for ops, cand in self.done_best.items():
            entry = DoneEntry(cand.ops, cand.words, cand.score, cand.sources)
            prev = merged.get(ops)
            if prev is None or entry.sort_key() < prev.sort_key():
                merged[ops] = entry
        return merged

    def done_winners(self, carried: Sequence[DoneEntry]) -> List[DoneEntry]:
        merged = sorted(
            self.done_identities(carried).values(), key=DoneEntry.sort_key
        )
        return merged[: self.done_capacity]


class SegmentedTopKSession:
    """可重复调用的分段 TopK 解码算子。"""

    def __init__(self, grammar: WorkOrderGrammar) -> None:
        self.grammar = grammar
        self._round: Optional[_Round] = None
        self._carried_done: List[DoneEntry] = []
        self._watched: Dict[str, _UnseenDecl] = {}
        self._retired: Dict[str, _Fingerprint] = {}
        self._step = 0
        self._invalidated: Optional[str] = None
        self._last_outcome: Optional[RoundOutcome] = None

    # ------------------------------------------------------------------ 轮次

    def begin_round(
        self,
        beams: Sequence[Draft],
        remaining: int,
        beam_capacity: int,
        done_capacity: int,
        unseen: Optional[Mapping[str, Mapping]] = None,
    ) -> None:
        """开始一轮：接收当前草稿、剩余步数、两池容量与未到分片声明。"""
        self._ensure_healthy()
        if self._round is not None:
            raise RuntimeError("上一轮尚未定案，不能开始新轮次；可先 abort_round 放弃")
        if remaining < 1:
            raise ValueError("剩余解码步数包含本轮，至少为 1")
        if beam_capacity < 0 or done_capacity < 0:
            raise ValueError("池容量不能为负")
        decls = {
            name: _UnseenDecl(spec["words"], spec["upper"])
            for name, spec in (unseen or {}).items()
        }
        self._round = _Round(
            step=self._step,
            beams=list(beams),
            remaining=remaining,
            beam_capacity=beam_capacity,
            done_capacity=done_capacity,
            unseen=decls,
        )

    def abort_round(self) -> None:
        """放弃当前轮（例如分片资料不一致后等待重发）。"""
        self._round = None

    def feed(self, shard: str, rows: Iterable[Sequence]) -> None:
        """消费一个分片的 ``(来源草稿, 候选词, 定点增量)`` 行。

        - 已在 ``unseen`` 中声明的分片按声明校验词域与增量上界；
        - 同名分片重发且内容一致时是空操作，不会挤出其他草稿；
        - 超界或重发内容不一致抛出 :class:`ShardInconsistency`；若相关轮次已
          提前定案，会话随之失效。
        """
        self._ensure_healthy()
        if shard in self._watched:
            self._feed_watched(shard, rows)
            return
        if shard in self._retired:
            fingerprint = _Fingerprint()
            for row in rows:
                if len(row) != 3:
                    raise ShardInconsistency(f"分片 {shard!r} 行格式错误: {row!r}")
                fingerprint.add(
                    _normalize_identity(row[0]), row[1], row[2]
                )
            if self._retired[shard].matches(fingerprint):
                return  # 已核验过的分片重发且内容一致：空操作
            self._invalidated = f"分片 {shard!r} 重发内容与已核验版本不一致"
            raise ShardInconsistency(self._invalidated)
        rnd = self._round
        if rnd is None:
            raise RuntimeError(f"当前没有进行中的轮次，无法消费分片 {shard!r}")
        fingerprint = _Fingerprint()
        staged = []
        for row in rows:
            if len(row) != 3:
                raise ValueError(f"分片行应为三元组: {row!r}")
            source = _normalize_identity(row[0])
            word = _check_word(row[1])
            delta = _check_score(row[2])
            fingerprint.add(source, word, delta)
            staged.append((source, word, delta))
        decl = rnd.pending.get(shard)
        if decl is not None:
            for _source, word, delta in staged:
                if word not in decl.words or delta > decl.upper:
                    raise ShardInconsistency(
                        f"分片 {shard!r} 超出声明范围（词域或增量上界），"
                        "依据它做的提前决定失效"
                    )
        previous = rnd.delivered.get(shard)
        if previous is not None:
            if previous.matches(fingerprint):
                return  # 重发内容一致：空操作
            raise ShardInconsistency(f"分片 {shard!r} 重发内容不一致")
        rnd.delivered[shard] = fingerprint
        rnd.pending.pop(shard, None)
        for source, word, delta in staged:
            self._consume_row(rnd, source, word, delta)

    def _feed_watched(self, shard: str, rows: Iterable[Sequence]) -> None:
        """定案之后迟到的受监视分片：只校验声明，不改动已定案结果。"""
        decl = self._watched.get(shard)
        if decl is None:
            raise RuntimeError(f"当前没有进行中的轮次，且分片 {shard!r} 不在监视中")
        fingerprint = _Fingerprint()
        for row in rows:
            if len(row) != 3:
                raise ShardInconsistency(f"分片 {shard!r} 行格式错误: {row!r}")
            source = _normalize_identity(row[0])
            word = row[1]
            delta = row[2]
            if (
                not isinstance(word, int)
                or isinstance(word, bool)
                or not isinstance(delta, int)
                or isinstance(delta, bool)
                or word not in decl.words
                or delta > decl.upper
            ):
                self._invalidated = f"分片 {shard!r} 超出声明范围，提前定案失效"
                raise ShardInconsistency(self._invalidated)
            fingerprint.add(source, word, delta)
        self._watched.pop(shard)
        self._retired[shard] = fingerprint

    def _consume_row(self, rnd: _Round, source: Identity, word: int, delta: int) -> None:
        beam = rnd.beams.get(source)
        if beam is None:
            return  # 来源草稿不在本轮席位中，不参与竞争
        transition = self.grammar.consume(beam.state, word)
        if transition is None:
            return  # 来源草稿不能消费该词
        next_state, op = transition
        words = beam.words + (word,)
        ops = beam.ops + ((op,) if op != "" else ())
        candidate = Candidate(
            words=words,
            ops=ops,
            state=next_state,
            score=beam.score + delta,
            source=beam.identity,
            sources=beam.sources,
        )
        steps_left = rnd.remaining - 1
        if self.grammar.outdegree(next_state) == 0:
            rnd.saw_grammar_blocked = True
        elif self.grammar.min_steps_to_accept(next_state) > steps_left:
            rnd.saw_budget_blocked = True
        if self.grammar.can_hold_seat(next_state, steps_left):
            seat = (ops, self.grammar.equivalence_of(next_state))
            occupant = rnd.seat_best.get(seat)
            if occupant is None or candidate.better_than(occupant):
                rnd.seat_best[seat] = candidate
        if self.grammar.is_accepting(next_state):
            occupant = rnd.done_best.get(ops)
            if occupant is None or candidate.better_than(occupant):
                rnd.done_best[ops] = candidate

    # ------------------------------------------------------------------ 定案

    def status(self) -> RoundStatus:
        """查询当前轮两池视图与定案状态（不改动任何状态）。"""
        self._ensure_healthy()
        rnd = self._require_round()
        impacts = self._unseen_impacts(rnd)
        finalized = not impacts
        continuation_entries = tuple(rnd.seat_winners())
        done_entries = tuple(rnd.done_winners(self._carried_done))
        continuation = PoolView(
            finalized=finalized,
            status=self._continuation_status(rnd, finalized, continuation_entries),
            entries=continuation_entries,
            pending_shards=tuple(sorted(rnd.pending)),
        )
        done = PoolView(
            finalized=finalized,
            status=self._done_status(rnd, finalized, done_entries),
            entries=done_entries,
            pending_shards=tuple(sorted(rnd.pending)),
        )
        return RoundStatus(
            step=rnd.step,
            remaining=rnd.remaining,
            continuation=continuation,
            done=done,
            delivered_shards=tuple(sorted(rnd.delivered)),
        )

    def finalize_round(self) -> RoundOutcome:
        """定案本轮：仅在两池均可宣布定案时成功。

        若仍有未到分片但已用分数上界与语法范围证明其不能改变席位，允许提前定案；
        这些分片转入监视，迟到时仍按声明校验，超界则使本次定案失效。
        """
        self._ensure_healthy()
        rnd = self._require_round()
        impacts = self._unseen_impacts(rnd)
        if impacts:
            raise PrematureFinalization(
                "仍有分片可能改写席位，不能定案: " + "; ".join(impacts)
            )
        next_beams = tuple(
            Draft(
                identity=cand.words,
                state=cand.state,
                words=cand.words,
                ops=cand.ops,
                score=cand.score,
                sources=cand.sources + (cand.words,),
            )
            for cand in rnd.seat_winners()
        )
        done_pool = tuple(rnd.done_winners(self._carried_done))
        self._carried_done = list(done_pool)
        self._watched.update(rnd.pending)
        outcome = RoundOutcome(
            step=rnd.step,
            next_beams=next_beams,
            done_pool=done_pool,
            watched_shards=tuple(sorted(rnd.pending)),
        )
        self._last_outcome = outcome
        self._round = None
        self._step += 1
        return outcome

    # ------------------------------------------------------------------ 查询

    def has_submittable(self) -> bool:
        """完成池中是否有可提交的有效完成命令。"""
        return bool(self._carried_done)

    def submit(self, identity: Optional[Ops] = None) -> Submission:
        """提交一份有效完成命令，默认取完成池最优；可回查词序列与来源草稿链。"""
        self._ensure_healthy()
        if not self._carried_done:
            raise RuntimeError("完成池为空，无可提交命令；请用 explain 查明原因")
        if identity is None:
            entry = self._carried_done[0]
        else:
            wanted = tuple(identity)
            matches = [e for e in self._carried_done if e.identity == wanted]
            if not matches:
                raise KeyError(f"完成池中没有该规范工单身份: {wanted!r}")
            entry = matches[0]
        return Submission(
            identity=entry.identity,
            command=entry.identity,
            words=entry.words,
            score=entry.score,
            sources=entry.sources,
        )

    def explain(self) -> Mapping:
        """说明当前会话：空池原因、是否预算用尽、是否仍有分片可能改写结果。"""
        report = {
            "invalidated": self._invalidated,
            "watched_shards": tuple(sorted(self._watched)),
            "done_pool_size": len(self._carried_done),
        }
        if self._round is not None:
            status = self.status()
            report["round"] = status
            report["continuation_empty_reason"] = (
                None
                if status.continuation.entries
                else status.continuation.status.value
            )
            report["done_empty_reason"] = (
                None if status.done.entries else status.done.status.value
            )
        elif not self._carried_done and self._last_outcome is not None:
            report["done_empty_reason"] = PoolStatus.EMPTY_NO_GRAMMAR_CANDIDATES.value
        return report

    @property
    def invalidated(self) -> bool:
        return self._invalidated is not None

    def scratch_report(self) -> Mapping:
        """局部缓冲规模：只按席位身份归约，不随词表大小保留全部扩展。"""
        if self._round is None:
            return {"seats": 0, "done_identities": 0, "beams": 0, "pending_shards": 0}
        rnd = self._round
        return {
            "seats": len(rnd.seat_best),
            "done_identities": len(rnd.done_best),
            "beams": len(rnd.beams),
            "pending_shards": len(rnd.pending),
        }

    # ------------------------------------------------------------------ 内部

    def _require_round(self) -> _Round:
        if self._round is None:
            raise RuntimeError("当前没有进行中的轮次")
        return self._round

    def _ensure_healthy(self) -> None:
        if self._invalidated is not None:
            raise InvalidatedSession(self._invalidated)

    def _continuation_status(
        self, rnd: _Round, finalized: bool, entries: Tuple
    ) -> PoolStatus:
        if entries:
            return PoolStatus.OK
        if not finalized:
            return PoolStatus.PENDING_SHARDS
        if rnd.beam_capacity == 0:
            return PoolStatus.EMPTY_BUDGET_EXHAUSTED
        # 定案且容量非负仍为空：能消费的候选要么剩余步内到不了合法终态
        # （预算用尽），要么落点在语法上不能再接词；否则它会作为新席位入池。
        if rnd.saw_budget_blocked or self._unseen_budget_blocked(rnd):
            return PoolStatus.EMPTY_BUDGET_EXHAUSTED
        return PoolStatus.EMPTY_NO_GRAMMAR_CANDIDATES

    def _done_status(self, rnd: _Round, finalized: bool, entries: Tuple) -> PoolStatus:
        if entries:
            return PoolStatus.OK
        if not finalized:
            return PoolStatus.PENDING_SHARDS
        if rnd.done_capacity == 0:
            return PoolStatus.EMPTY_BUDGET_EXHAUSTED
        # 完成身份只要求落在可提交状态；定案且容量非负仍为空，说明语法层面
        # 没有任何候选能到达可提交状态（否则它会作为新身份进入池中）。
        return PoolStatus.EMPTY_NO_GRAMMAR_CANDIDATES

    def _unseen_budget_blocked(self, rnd: _Round) -> bool:
        """未到分片中是否有"能消费但剩余步内到不了终态"的候选。"""
        steps_left = rnd.remaining - 1
        for decl in rnd.pending.values():
            for beam in rnd.beams.values():
                for word in decl.words:
                    transition = self.grammar.consume(beam.state, word)
                    if transition is None:
                        continue
                    next_state = transition[0]
                    if (
                        self.grammar.outdegree(next_state) > 0
                        and self.grammar.min_steps_to_accept(next_state) > steps_left
                    ):
                        return True
        return False

    def _unseen_impacts(self, rnd: _Round) -> List[str]:
        """列出未到分片仍可能改写池内容的具体理由；为空则相应池可宣布定案。

        对每份未到分片，用词域与增量上界构造"最强可能候选"（分数取上界），
        与当前池内容按偏好序精确比较：上界严格更优才可能改写；上界与现任者
        同分时按完整词 ID 序列与来源 ID 的次序判断——纯分数界不能把未决路径
        提前排除。
        """
        impacts: List[str] = []
        if not rnd.pending:
            return impacts
        steps_left = rnd.remaining - 1
        seat_winners = rnd.seat_winners()
        in_pool_seats = {
            (c.ops, self.grammar.equivalence_of(c.state)) for c in seat_winners
        }
        worst_seat_key = seat_winners[-1].sort_key() if seat_winners else None
        seats_full = len(seat_winners) >= rnd.beam_capacity
        done_winners = rnd.done_winners(self._carried_done)
        in_pool_done = {entry.identity for entry in done_winners}
        worst_done_key = done_winners[-1].sort_key() if done_winners else None
        done_full = len(done_winners) >= rnd.done_capacity
        done_all = rnd.done_identities(self._carried_done)
        for name, decl in sorted(rnd.pending.items()):
            for beam in rnd.beams.values():
                for word in decl.words:
                    transition = self.grammar.consume(beam.state, word)
                    if transition is None:
                        continue
                    next_state, op = transition
                    bound = beam.score + decl.upper
                    challenger = Candidate(
                        words=beam.words + (word,),
                        ops=beam.ops + ((op,) if op != "" else ()),
                        state=next_state,
                        score=bound,
                        source=beam.identity,
                        sources=beam.sources,
                    )
                    key = challenger.sort_key()
                    if rnd.beam_capacity > 0 and self.grammar.can_hold_seat(
                        next_state, steps_left
                    ):
                        seat = (
                            challenger.ops,
                            self.grammar.equivalence_of(next_state),
                        )
                        occupant = rnd.seat_best.get(seat)
                        if occupant is None:
                            if not seats_full or (
                                worst_seat_key is not None and key < worst_seat_key
                            ):
                                impacts.append(
                                    f"分片 {name!r} 的词 {word} 可能新增续写席位"
                                )
                        elif seat in in_pool_seats:
                            if key < occupant.sort_key():
                                impacts.append(
                                    f"分片 {name!r} 的词 {word} 可能改写续写席位"
                                )
                        elif key < occupant.sort_key() and (
                            worst_seat_key is not None and key < worst_seat_key
                        ):
                            # 刷新池外席位并凭新实参挤入池中
                            impacts.append(
                                f"分片 {name!r} 的词 {word} 可能挤入续写池"
                            )
                    if rnd.done_capacity > 0 and self.grammar.is_accepting(next_state):
                        entry = DoneEntry(
                            challenger.ops, challenger.words, bound, beam.sources
                        )
                        entry_key = entry.sort_key()
                        occupant = done_all.get(challenger.ops)
                        if occupant is None:
                            if not done_full or (
                                worst_done_key is not None
                                and entry_key < worst_done_key
                            ):
                                impacts.append(
                                    f"分片 {name!r} 的词 {word} 可能新增完成身份"
                                )
                        elif challenger.ops in in_pool_done:
                            if entry_key < occupant.sort_key():
                                impacts.append(
                                    f"分片 {name!r} 的词 {word} 可能刷新完成身份"
                                )
                        elif entry_key < occupant.sort_key() and (
                            worst_done_key is not None
                            and entry_key < worst_done_key
                        ):
                            impacts.append(
                                f"分片 {name!r} 的词 {word} 可能挤入完成池"
                            )
        return impacts
