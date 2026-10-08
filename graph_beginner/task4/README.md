# 任务四：知识图谱补全

使用 PyTorch 实现 TransE、RotatE 和 ConvE，在 Nations 数据集上完成知识图谱补全。训练采用反向关系和负采样，测试采用过滤排名，指标包括 MRR、MR 和 Hits@1/3/10。

## 数据划分

Nations 提供独立的训练集、验证集和测试集。训练集用于更新参数，验证集用于选择最优轮次，测试集只用于最终评估。过滤排名会排除训练、验证和测试集中已经成立的其他三元组。

## 运行

```powershell
cd task4
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python code/data.py
.venv\Scripts\python code/test_task4.py
```

运行三个模型的基准实验：

```powershell
.venv\Scripts\python code/benchmark.py --epochs 100 --device cpu `
  --output results/baseline
.venv\Scripts\python code/audit_results.py --input results/baseline
```

运行 ConvE 学习率和卷积层数实验：

```powershell
.venv\Scripts\python code/benchmark.py --models conve `
  --lrs 0.001,0.005,0.01 --layers 1,2,3 --epochs 100 `
  --device cpu --in-process --output results/ablation
.venv\Scripts\python code/audit_results.py --input results/ablation `
  --models conve --lrs 0.001,0.005,0.01 --layers 1,2,3 --epochs 100
```

单次训练与复测：

```powershell
.venv\Scripts\python code/train.py --model rotate --output results/single
.venv\Scripts\python code/evaluate.py results/single/best.pt
```

实验结果见 [REPORT.md](REPORT.md)。
