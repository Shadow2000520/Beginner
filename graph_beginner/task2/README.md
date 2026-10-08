# 任务二：链路预测

使用 PyTorch Geometric 在 Cora、CiteSeer、Flickr 上测试 GCN、GAT、GraphSAGE 和 GIN，并比较全图训练与邻居采样训练。

## 环境

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt --find-links https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

## 数据

```powershell
.venv/Scripts/python.exe code/data.py
```

每个数据集按 85%/5%/10% 划分训练、验证和测试边。训练监督边与消息传递边分离，负样本从不存在的边中采样。批量实验最多使用 20,000 条训练正边。

## 训练和测试

```powershell
# 全图训练
.venv/Scripts/python.exe code/train.py --dataset Cora --model gcn --mode full --epochs 30 --output results/example_full

# 邻居采样训练
.venv/Scripts/python.exe code/train.py --dataset Cora --model gcn --mode sampled --neighbors 15 10 --epochs 30 --output results/example_sampled

# 测试保存的模型
.venv/Scripts/python.exe code/evaluate.py --checkpoint results/example_full/best.pt
```

## 批量实验

```powershell
# 3 个数据集、4 种模型、2 种训练方式
.venv/Scripts/python.exe code/benchmark.py --epochs 30 --device cuda --output results/baseline --resume

# 学习率和层数实验
.venv/Scripts/python.exe code/benchmark.py --datasets Cora --lrs 0.001 0.01 0.05 --layers 1 2 3 --epochs 30 --device cuda --output results/ablation --resume --in-process

# 结果检查
.venv/Scripts/python.exe code/audit_results.py --input results/baseline
```

训练结果包括 AUC、Average Precision、Accuracy、运行时间、训练记录和最佳模型。
