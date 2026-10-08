# 任务三：图分类

使用 PyTorch Geometric 在 MUTAG 和 ZINC 上比较 GCN、GAT、GraphSAGE、GIN，以及 mean、max、min 三种全局池化方式。

## 数据处理

- MUTAG：按类别分层划分为训练集、验证集和测试集，比例约为 8:1:1。
- ZINC：其原始目标是连续值。这里用训练集目标的十分位点划分 10 个类别，验证集和测试集沿用训练集边界。基准实验随机取 3000 个训练图，官方验证集和测试集保持完整。

## 运行

```powershell
cd task3
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python code/data.py
.venv\Scripts\python code/test_task3.py
```

运行模型与池化对比：

```powershell
.venv\Scripts\python code/benchmark.py --epochs 30 --output results/baseline
.venv\Scripts\python code/audit_results.py --input results/baseline
```

运行学习率和层数实验：

```powershell
.venv\Scripts\python code/benchmark.py --datasets MUTAG --poolings mean `
  --lrs 0.001,0.005,0.01 --layers 2,3,4 --max-train-graphs 0 `
  --epochs 30 --in-process --output results/ablation
.venv\Scripts\python code/audit_results.py --input results/ablation `
  --datasets MUTAG --poolings mean --lrs 0.001,0.005,0.01 `
  --layers 2,3,4 --epochs 30
```

运行全批次实验，并与基准实验中的分批训练比较：

```powershell
.venv\Scripts\python code/benchmark.py --poolings mean --batch-size 100000 `
  --epochs 30 --output results/full_batch
.venv\Scripts\python code/audit_results.py --input results/full_batch `
  --poolings mean --epochs 30
.venv\Scripts\python code/batch_compare.py --mini results/baseline `
  --full results/full_batch --output results/batch_comparison
```

单次训练与复测：

```powershell
.venv\Scripts\python code/train.py --dataset MUTAG --model gin --pooling mean `
  --output results/single
.venv\Scripts\python code/evaluate.py results/single/best.pt
```

实验结果见 [REPORT.md](REPORT.md)。
