"""资料交接：把采集记录接到分段 TopK 算子上，并支持回放多轮解码。

模型解码侧的接入方式：

- 用 :func:`grammar_from_records` / :func:`drafts_from_records` 把交接记录
  转成算子所需的语法与草稿；
- 每一轮构造 :class:`RoundPlan`（剩余步数、两池容量、未到分片声明），模型每
  导出一个打分块就调用一次 :meth:`SegmentedTopKSession.feed`；
- 用 :meth:`SegmentedTopKSession.status` 查询两池是否已定案，定案后
  :meth:`SegmentedTopKSession.finalize_round` 取出下一轮续写身份；
- 全部轮次结束后 :meth:`SegmentedTopKSession.submit` 提交有效完成命令，
  并可回查它来自哪些词和哪些草稿。

:func:`replay` 按采集记录中的轮次与分片顺序完整回放，返回结构化报告；
``python -m segbeam.adapter data/capture.json`` 打印同一过程的可读轨迹。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from caseio import load_case

from .grammar import WorkOrderGrammar
from .operator import (
    Draft,
    PoolStatus,
    RoundStatus,
    RoundOutcome,
    SegmentedTopKSession,
    Submission,
)


def grammar_from_records(records: Mapping) -> WorkOrderGrammar:
    """从交接记录的 ``grammar`` 字段构造正式工单语法。"""
    return WorkOrderGrammar.from_record(records["grammar"])


def drafts_from_records(beams: Sequence[Mapping]) -> List[Draft]:
    """把交接记录中的草稿行转成算子草稿（首轮使用给定身份）。"""
    return [
        Draft(
            identity=row["id"],
            state=row["state"],
            words=tuple(row["words"]),
            ops=tuple(row["ops"]),
            score=int(row["score"]),
        )
        for row in beams
    ]


@dataclass(frozen=True)
class RoundPlan:
    """现场设定的一轮参数与未到分片声明。"""

    remaining: int
    beam_capacity: int
    done_capacity: int
    unseen: Mapping[str, Mapping] = field(default_factory=dict)


@dataclass(frozen=True)
class FeedTrace:
    """一次分片消费后的状态快照。"""

    shard: str
    status: RoundStatus


@dataclass(frozen=True)
class RoundReport:
    """一轮的回放报告：每次分片消费后的状态与最终定案输出。"""

    plan: RoundPlan
    feeds: Tuple[FeedTrace, ...]
    outcome: RoundOutcome


@dataclass(frozen=True)
class ReplayReport:
    """完整回放报告：各轮轨迹、最终提交与来源、会话说明。"""

    rounds: Tuple[RoundReport, ...]
    submission: Optional[Submission]
    explanation: Mapping


def run_round(
    session: SegmentedTopKSession,
    beams: Sequence[Draft],
    plan: RoundPlan,
    shards: Iterable[Tuple[str, Iterable[Sequence]]],
) -> RoundReport:
    """驱动一轮解码：按分片到达顺序消费，逐个记录定案状态后定案。

    ``shards`` 是 ``(分片名, 行迭代器)`` 的有序序列，对应模型一次导出中
    按词表切分的打分块。返回本轮报告；定案输出供下一轮继续使用。
    """
    session.begin_round(
        beams,
        remaining=plan.remaining,
        beam_capacity=plan.beam_capacity,
        done_capacity=plan.done_capacity,
        unseen=plan.unseen,
    )
    feeds: List[FeedTrace] = []
    for name, rows in shards:
        session.feed(name, rows)
        feeds.append(FeedTrace(shard=name, status=session.status()))
    outcome = session.finalize_round()
    return RoundReport(plan=plan, feeds=tuple(feeds), outcome=outcome)


def replay(path: Path) -> ReplayReport:
    """按采集记录回放多轮解码，返回结构化报告。"""
    records, _arrays = load_case(Path(path))
    grammar = grammar_from_records(records)
    session = SegmentedTopKSession(grammar)
    beams: List[Draft] = []
    reports: List[RoundReport] = []
    for index, round_record in enumerate(records["rounds"]):
        if index == 0:
            beams = drafts_from_records(round_record["beams"])
        plan = RoundPlan(
            remaining=int(round_record["remaining"]),
            beam_capacity=int(round_record["beam_capacity"]),
            done_capacity=int(round_record["done_capacity"]),
            unseen=round_record.get("unseen", {}),
        )
        shards = [
            (name, [tuple(row) for row in rows])
            for name, rows in round_record["scores"].items()
        ]
        report = run_round(session, beams, plan, shards)
        reports.append(report)
        beams = list(report.outcome.next_beams)
    submission = session.submit() if session.has_submittable() else None
    return ReplayReport(
        rounds=tuple(reports),
        submission=submission,
        explanation=session.explain(),
    )


def _format_pool(entries) -> str:
    parts = []
    for entry in entries:
        words = list(entry.words)
        ops = list(entry.ops) if hasattr(entry, "ops") else list(entry.identity)
        parts.append(f"词{words} 操作{ops} 分{entry.score}")
    return "；".join(parts) if parts else "（空）"


def main() -> None:
    parser = argparse.ArgumentParser(description="回放工单语法约束下的分段 TopK 解码")
    parser.add_argument("path", type=Path)
    settings = parser.parse_args()
    report = replay(settings.path)
    for round_report in report.rounds:
        plan = round_report.plan
        print(
            f"== 第 {round_report.outcome.step} 轮 剩余步数 {plan.remaining} "
            f"续写容量 {plan.beam_capacity} 完成容量 {plan.done_capacity}"
        )
        for feed in round_report.feeds:
            status = feed.status
            mark = lambda view: "定案" if view.finalized else f"待分片 {list(view.pending_shards)}"
            print(
                f"  分片 {feed.shard} 到达：续写池[{mark(status.continuation)}] "
                f"{_format_pool(status.continuation.entries)}"
            )
            print(
                f"               完成池[{mark(status.done)}] "
                f"{_format_pool(status.done.entries)}"
            )
        outcome = round_report.outcome
        print(f"  定案：续写身份 {[list(b.identity) for b in outcome.next_beams]}")
        if outcome.watched_shards:
            print(f"  提前定案，转入监视的分片: {list(outcome.watched_shards)}")
    if report.submission is not None:
        sub = report.submission
        print("== 提交有效完成命令")
        print(f"  规范命令: {list(sub.command)}  分数 {sub.score}")
        print(f"  来自词序列: {list(sub.words)}")
        print(f"  来自草稿链: {[list(s) if isinstance(s, tuple) else s for s in sub.sources]}")
    else:
        print("== 完成池为空，无法提交")
    print(f"== 会话说明: {json.dumps(_jsonable(report.explanation), ensure_ascii=False)}")


def _jsonable(value):
    if isinstance(value, Mapping):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (RoundStatus,)):
        return repr(value)
    if isinstance(value, PoolStatus):
        return value.value
    return value


if __name__ == "__main__":
    main()
