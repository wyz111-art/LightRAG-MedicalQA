# 医疗问答正式评测集

`benchmark_v1.json` 是本项目的第一版固定评测集，覆盖 32 个问题、6 类医疗
问答意图和多个科室。评测不会修改知识库。

## 快速验证数据集

```powershell
.\.venv\Scripts\python.exe .\medical_eval.py --validate-only
```

## 检索评测

只运行意图识别、实体提取、路由和检索，不调用最终回答模型：

```powershell
.\.venv\Scripts\python.exe .\medical_eval.py --stage retrieval
```

先抽查 5 条：

```powershell
.\.venv\Scripts\python.exe .\medical_eval.py --stage retrieval --limit 5
```

运行指定用例：

```powershell
.\.venv\Scripts\python.exe .\medical_eval.py `
  --stage retrieval `
  --case-id htn-symptoms `
  --case-id htn-treatment
```

## 端到端回答评测

端到端模式还会检查答案是否包含预期事实、是否引用有效证据：

```powershell
.\.venv\Scripts\python.exe .\medical_eval.py --stage answer
```

本地 CPU 模型运行完整回答评测可能耗时较长，建议先使用 `--limit`。

## 报告

报告写入 `medical_evaluation/reports/`：

- JSON：保存每项检查的期望值和实际值，适合后续分析。
- Markdown：汇总通过率、各项指标准确率、延迟和失败用例。

可使用阈值让低于要求的评测返回非零退出码：

```powershell
.\.venv\Scripts\python.exe .\medical_eval.py `
  --stage retrieval `
  --fail-below 0.80
```

## 新增用例要求

每条用例至少包含：

- 稳定且唯一的 `id`；
- 用户问题 `question`；
- 预期意图和实体；
- 适用时的关系类型与至少一个可接受的图谱目标；
- 最少图谱或文本证据数量；
- 端到端评测时可接受的答案关键词。

修改数据、Prompt、模型或检索策略后，应使用同一版本评测集重新运行，避免只
根据个别演示问题判断效果。
