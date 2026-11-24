# 工单语法约束下的分段 Beam TopK

工艺命令字典使用整数词号。切分的打分块来自同一轮模型导出。 数据为本地合成。

- `data/capture.json` 保存采集原始内容。records 是交换记录，tensors 字段分开描述名称、类型、形状和值。整数索引从零开始，未指定闭合的区间均为左闭右开。
- `data/capacity.json` 保存制造端的容量、规模或分片条件。大尺寸是接口需考虑的使用规模。工作区计入新分配的解包、转换、归约、版本暂存及交接缓冲。调用方已有只读输入和已消费的输出不计入工作区，额外物化副本计入。

## 交换约定

`transitions` 每行是起始状态、词 ID、相继状态和应追加的规范操作 ID，空操作用空字符串。各状态的 `equivalence` 由语法发布者给出。`accepting` 为可提交状态，可继续接词的合法终态也可竞争续写席位。步数包括本轮。已定案的完成池沿轮次带入并与新候选合并。
`beams` 中词历史和规范操作是两份独立身份信息。`scores` 每行是来源草稿 ID、候选词、定点分数增量。从第二轮开始，来源身份使用完整词 ID 序列（数组）。`unseen` 的每项提供每份相应分片的可能词域和增量上界。分片若超界，依据它做的提前决定失效；该轮须报资料不一致，后续草稿不能消费错误定案。

## 读取资料

使用 Python 3.11。首次创建本地环境时安装锁定依赖。只读查看显示原始字段和张量尺寸。

```sh
python -m pip install -r requirements.txt
python caseio.py data/capture.json
python -m unittest discover -s tests -v
```

现有检查核对采集格式和所附数据的结构。

## 分段 TopK 解码算子

`segbeam` 把多轮解码做成可在 Python 解码侧重复调用的能力，直接接到模型导出循环上：

```python
from segbeam import SegmentedTopKSession, WorkOrderGrammar
from segbeam.adapter import RoundPlan, run_round

grammar = WorkOrderGrammar.from_record(records["grammar"])
session = SegmentedTopKSession(grammar)
beams = drafts                       # 带累积分、稳定身份、解析位置的当前草稿
for round_record in rounds:          # 现场设定剩余步数与两池容量
    plan = RoundPlan(remaining=..., beam_capacity=..., done_capacity=...,
                     unseen=round_record["unseen"])
    report = run_round(session, beams, plan, model_shards)  # 模型每导出一个打分块 feed 一次
    beams = report.outcome.next_beams                       # 已定案的续写身份
submission = session.submit()        # 提交有效完成命令，可回查词序列与来源草稿链
```

- **席位竞争**：候选只在其来源草稿能消费该词时参与竞争。同一规范操作序列且同一
  语法状态等价类的候选竞争同一席位，同义写法不会挤掉本可续写的草稿；显示字符串
  相同但后续合法范围不同的候选保留各自身份。偏好序为累积分高者优先，同分按完整
  词 ID 序列字典序，再按来源 ID。
- **两池**：能提前终止的命令进入完成池，语法允许继续接词时也占一个续写席位；
  剩余步内到不了合法终态的候选不能占续写席位。以前各轮已定案的完成项继续参加
  完成池竞争；完成池按完整规范工单身份归并，由较优实参路径提供可追查的词序列。
- **定案**：分片全部到齐，或用 `unseen` 的可信分数上界与语法范围证明剩余分片
  不能改变席位时，相应池才定案；同分时纯分数界不能提前排除未决路径。提前定案
  所依赖的分片转入监视，迟到时仍按声明校验；超界或重发内容不一致报
  `ShardInconsistency`，已提前定案的会话随之失效，后续草稿不能消费错误定案。
  分片重发内容一致时是空操作，不会挤出其他草稿。
- **缓冲**：局部缓冲只按席位身份归约候选，规模受语法（操作数 × 等价类数）约束，
  不随词表大小保留全部扩展；`session.scratch_report()` 可查看当前规模。
- **查询**：`session.status()` 报告两池视图与未到分片；`session.explain()` 区分
  空池是语法层面无候选、预算用尽，还是仍有分片可能改写结果。

回放采集记录并打印每轮定案轨迹：

```sh
python -m segbeam.adapter data/capture.json
```
