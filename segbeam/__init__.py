"""工单语法约束下的分段 TopK 解码算子。

把"模型按词表分片回传分数"的多轮解码做成可重复调用的能力：

- :class:`WorkOrderGrammar` 描述正式工单语法（转移、规范操作、等价类、可提交状态）。
- :class:`SegmentedTopKSession` 是多轮解码算子：每轮接收带累积分、稳定身份和
  解析位置的草稿，按分片增量消费候选分数，维护续写池与完成池，并给出定案状态。
- 候选只在其来源草稿能消费该词时参与竞争；同一规范操作序列且同一语法状态等价类
  的候选竞争同一席位；得分是沿路径累加的定点整数，高分优先，同分按完整词 ID
  序列字典序、再按来源 ID 排序。
- 仅当相关分片全部到齐，或可信分数上界与语法范围证明剩余分片不能改变席位时，
  相应池才宣布定案；分片超界会使依据它做的提前定案失效。
"""
from .grammar import WorkOrderGrammar
from .operator import (
    Candidate,
    DoneEntry,
    Draft,
    InvalidatedSession,
    PoolStatus,
    PoolView,
    PrematureFinalization,
    RoundOutcome,
    RoundStatus,
    SegmentedTopKSession,
    ShardInconsistency,
    Submission,
)

__all__ = [
    "Candidate",
    "DoneEntry",
    "Draft",
    "InvalidatedSession",
    "PoolStatus",
    "PoolView",
    "PrematureFinalization",
    "RoundOutcome",
    "RoundStatus",
    "SegmentedTopKSession",
    "ShardInconsistency",
    "Submission",
    "WorkOrderGrammar",
]
