# 任务一：节点分类

使用 PyTorch Geometric 在 Cora、CiteSeer、Flickr 上训练 GCN、GAT、GraphSAGE 和 GIN，并比较全图训练与邻居采样训练。

## 目录

```text
task1/
  code/          训练、测试和实验脚本
  data/          数据集及统计信息
  results/       实验结果和图表
  README.md      运行说明
  REPORT.md      实验报告
  requirements.txt
```

## 环境

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt --find-links https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

CPU 环境将两个链接中的 `cu128` 改为 `cpu`。

## 数据

```powershell
.venv/Scripts/python.exe code/data.py
```

Cora 和 CiteSeer 使用 Planetoid public 划分，Flickr 使用数据集自带划分。数据统计保存在 `data/statistics.json`。

## 训练和测试

```powershell
# 全图训练
.venv/Scripts/python.exe code/train.py --dataset Cora --model gcn --mode full --epochs 100 --output results/example_full

# 邻居采样训练
.venv/Scripts/python.exe code/train.py --dataset Cora --model gcn --mode sampled --neighbors 15 10 --batch-size 64 --epochs 100 --output results/example_sampled

# 测试保存的模型
.venv/Scripts/python.exe code/evaluate.py --checkpoint results/example_full/best.pt
```

可选模型：`gcn`、`gat`、`sage`、`gin`；可选数据集：`Cora`、`CiteSeer`、`Flickr`。主要参数包括 `--lr`、`--layers`、`--hidden`、`--dropout`、`--batch-size` 和 `--neighbors`。

## 批量实验

```powershell
# 模型与训练方式对比
.venv/Scripts/python.exe code/benchmark.py --epochs 100 --seeds 42 43 44 --device cuda --output results/baseline --resume

# 学习率与层数实验
.venv/Scripts/python.exe code/benchmark.py --datasets Cora --lrs 0.001 0.01 0.05 --layers 1 2 3 --seeds 42 --epochs 100 --device cuda --output results/ablation --resume --in-process

# 重新生成汇总
.venv/Scripts/python.exe code/report.py --input results/baseline
```

每次训练会生成 `config.json`、`history.csv`、`metrics.json` 和 `best.pt`。批量实验另外生成 `summary.csv`、`summary.md` 和对比图。

## 检查

```powershell
.venv/Scripts/python.exe -m unittest discover -s code -p "test_*.py" -v
.venv/Scripts/python.exe code/audit_results.py --input results/baseline
```

## 参考

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/2.7.0/)
- [NeighborLoader](https://pytorch-geometric.readthedocs.io/en/2.7.0/tutorial/neighbor_loader.html)
