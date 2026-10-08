# 实验结果

- `baseline/`：模型与池化方式对比。
- `ablation/`：学习率与网络层数实验。
- `full_batch/`：全部训练图组成一个批次的实验。
- `batch_comparison/`：全批次和分批训练的对比。

每次运行保存配置、逐轮训练记录、指标和最优模型。汇总脚本生成 CSV、Markdown 表格和图表。
