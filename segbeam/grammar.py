"""正式工单语法：转移、规范操作、状态等价类与可达性。

语法发布方给出：

- ``transitions``：每行是 ``[起始状态, 词 ID, 相继状态, 规范操作 ID]``，空操作用空字符串。
- ``equivalence``：各状态的等价类。后续合法范围（可接受的后续语言）不同的状态
  属于不同等价类；即使两份草稿的规范操作序列相同，落在不同等价类也要保留各自身份。
- ``accepting``：可提交状态。可继续接词的合法终态也可竞争续写席位。
"""
from __future__ import annotations

from collections import deque
from math import inf
from typing import Dict, Hashable, Iterable, Mapping, Tuple

State = Hashable


class WorkOrderGrammar:
    """工单语法的不可变模型，供解码侧反复查询。"""

    def __init__(
        self,
        transitions: Iterable[tuple],
        accepting: Iterable[State],
        equivalence: Mapping[State, Hashable],
    ) -> None:
        self._consume: Dict[State, Dict[int, Tuple[State, str]]] = {}
        self._equivalence: Dict[State, Hashable] = dict(equivalence)
        self._accepting = frozenset(accepting)
        states = set(self._equivalence) | set(self._accepting)
        for row in transitions:
            if len(row) != 4:
                raise ValueError(f"转移行应为四元组: {row!r}")
            src, word, dst, op = row
            if not isinstance(word, int) or isinstance(word, bool):
                raise ValueError(f"词 ID 应为整数: {row!r}")
            if not isinstance(op, str):
                raise ValueError(f"规范操作 ID 应为字符串: {row!r}")
            states.add(src)
            states.add(dst)
            slot = self._consume.setdefault(src, {})
            if word in slot:
                raise ValueError(f"状态 {src!r} 对词 {word} 的转移重复定义")
            slot[word] = (dst, op)
        missing = states - set(self._equivalence)
        if missing:
            raise ValueError(f"以下状态缺少发布方等价类: {sorted(missing)!r}")
        self._states = frozenset(states)
        # 反向 BFS 求每个状态到最近可提交状态的最少步数，用于剩余步数可行性判定。
        reverse: Dict[State, list] = {s: [] for s in self._states}
        for src, table in self._consume.items():
            for dst, _op in table.values():
                reverse[dst].append(src)
        self._min_steps: Dict[State, float] = {s: inf for s in self._states}
        queue = deque()
        for state in self._accepting:
            self._min_steps[state] = 0
            queue.append(state)
        while queue:
            state = queue.popleft()
            for prev in reverse[state]:
                if self._min_steps[prev] == inf:
                    self._min_steps[prev] = self._min_steps[state] + 1
                    queue.append(prev)

    @classmethod
    def from_record(cls, record: Mapping) -> "WorkOrderGrammar":
        """从交接记录中的 ``grammar`` 字段构造。"""
        return cls(
            transitions=[tuple(row) for row in record["transitions"]],
            accepting=record["accepting"],
            equivalence=record["equivalence"],
        )

    @property
    def states(self) -> frozenset:
        return self._states

    def consume(self, state: State, word: int) -> Tuple[State, str] | None:
        """状态 ``state`` 消费词 ``word``，返回 ``(相继状态, 规范操作)``；不能消费返回 None。"""
        table = self._consume.get(state)
        if table is None:
            return None
        return table.get(word)

    def equivalence_of(self, state: State) -> Hashable:
        return self._equivalence[state]

    def is_accepting(self, state: State) -> bool:
        return state in self._accepting

    def outdegree(self, state: State) -> int:
        table = self._consume.get(state)
        return len(table) if table else 0

    def min_steps_to_accept(self, state: State) -> float:
        """从 ``state`` 到最近可提交状态的最少转移步数，不可达为正无穷。"""
        return self._min_steps[state]

    def can_hold_seat(self, state: State, steps_left: int) -> bool:
        """判断候选在消费本轮之后能否占续写席位。

        ``steps_left`` 是本轮之后剩余的解码步数。要求：

        - 在剩余步内能到达合法终态（``min_steps_to_accept(state) <= steps_left``）；
        - 语法允许继续接词（有出边）。可提前终止但不能再接词的终态不占续写席位，
          它只进入完成池。
        """
        if steps_left < 0:
            return False
        return self._min_steps[state] <= steps_left and self.outdegree(state) > 0
