# 任务二：从零实现 mini-GPT

> 主大纲见仓库根 [README](../README.md)；本目录是该任务的资源、自检与提交入口。

## 一句话目标

用 PyTorch 从零搭一个 decoder-only mini-GPT：手写 BPE 分词、集成 RoPE、实现 KV cache，在中文小语料上预训练到困惑度达标，并能自回归生成连贯文本。**扩展**实践书 v2「nanoGPT 模型」的带读——nanoGPT 用绝对位置编码、只讲不实现 KV cache，这里都亲手补上。

## 任务情境

假装你要给团队做一个「最小可用」的语言模型底座，组长的要求是：

- 不许用 `tiktoken` / `sentencepiece`，分词器自己写
- 位置编码要用 RoPE，不要绝对位置编码
- 推理要带 KV cache，能讲清楚它省了什么
- 三周后汇报：困惑度曲线 + 几段生成样例 + 你对 BPE、RoPE、KV cache 的理解

这就是本任务。

## 输入 / 输出

| | 内容 |
|---|---|
| **给你** | 三档语料（唐诗 ~49KB quick-start / TinyStories / SkyPile 子集，`data/download.py` 自动准备）/ PyTorch 2.7+ / 唐诗档 CPU 即可，TinyStories CPU 可训但慢、建议 GPU，SkyPile 子集建议 GPU |
| **交付** | 1. `ckpt/tokenizer.json`（训练好的 BPE 词表） 2. `ckpt/best.pt`（预训练模型 state_dict） 3. 几段生成样例（不同采样策略对比） 4. `eval/result.json`（自检结果） 5. 一段 200–500 字实验观察 |

## Definition of Done

必做 5 项，缺一不算完成：

- [ ] **M1** 手写简化版 BPE tokenizer，自检 `tokenizer_roundtrip` 通过（encode→decode 能还原中文）
- [ ] **M2** 手写 decoder-only 模型并集成 **RoPE**，前向不报错且形状对
- [ ] **M3** 实现 **KV cache**，自检 `kv_cache_equivalence` 通过（开/关 cache logits 误差 < 1e-4）
- [ ] **M4** 在语料上预训练，自检 `perplexity_on_dev` 低于阈值（唐诗 < 50 / TinyStories < 10）
- [ ] **M5** 实现 greedy / top-k / top-p / temperature 四种采样，能生成连贯文本

加分（任选）：

- [ ] **S1** 参数量扫描（10M / 50M / 100M）vs 困惑度
- [ ] **S2** 绝对位置编码 vs RoPE 在长序列外推上的差异
- [ ] **S3** KV cache 开 / 关的推理速度对比（同样词元数计时）
- [ ] **S4** TinyStories 上复现 10M 参数模型涌现叙事能力

## 实施步骤（建议节奏：3 周）

### 第 1-2 天：环境 + 数据

```bash
pip install -r requirements.txt

# 三档数据，按设备和目标选；不加参数默认 poetry，适合先跑通
python data/download.py                         # ~49KB 唐诗 quick-start
python data/download.py --dataset poetry        # ~49KB，CPU 即可，5 分钟跑通
python data/download.py --dataset tinystories   # 英文故事语料，CPU 可训，看小模型叙事能力
python data/download.py --dataset skypile       # ~1GB+，建议 GPU
```

脚本统一生成 `data/train.txt`、`data/dev.txt`、`data/dataset_info.json`。训练代码读 `train.txt`，自检用 `dev.txt` 算困惑度；自行换中文语料也请保持这两个文件名。

### 第 3-6 天：BPE tokenizer（M1）

**输入**：`train.txt` 语料
**输出**：`src/tokenizer.py` 完整，`ckpt/tokenizer.json` 词表，能通过 `tokenizer_roundtrip` 自检

实现 `class BPETokenizer`：统计相邻 token 对频率、迭代 merge、导出词表，`encode`/`decode`/`from_pretrained` 齐全。

**常见坑**：

- 字节级（byte-level）还是字符级没想清楚：中文若按字符级，未登录字会丢；按字节级要保证 `decode` 能跨 UTF-8 多字节边界还原
- merge 顺序错：合并必须每轮取当前频率最高的对，且 `decode` 时按训练时的 merge 表反向还原
- `vocab_size` 没暴露，下游建 embedding 时对不上

### 第 7-12 天：模型 + RoPE + KV cache（M2 + M3）

**输入**：tokenizer 编码后的 id 序列
**输出**：`src/model.py` 完整，通过 `kv_cache_equivalence` 自检

实现内容：

1. `src/rope.py`：旋转位置编码，作用在 Q/K 上
2. `src/attention.py`：causal multi-head attention，支持传入并更新 KV cache
3. `src/model.py`：`class MiniGPT`，`forward(ids, kv_cache=None, return_cache=False)` + `generate(...)`

**常见坑**：

- RoPE 维度配对方式（相邻两维一组 vs 前后折半）实现与解码要一致；频率基底常用 `base=10000`
- RoPE 只加在 Q/K，**不要**加在 V
- KV cache 增量解码时，新词元的 position 要接着历史长度算（否则 RoPE 角度错、和全量前向对不上，`kv_cache_equivalence` 直接挂）
- cache 拼接维度搞反（应在序列长度 T 维上 append）

### 第 13-16 天：训练（M4）

本地 M2/M3 实现可先独立验证，无需训练权重：

```bash
python eval/test_kv_cache_equivalence.py
python -m unittest discover -s tests -v
```

`MiniGPT` 使用 Pre-LN、GELU 前馈网络和相邻维度配对的 RoPE。输入形状为
`[B, T]`，输出 logits 为 `[B, T, vocab_size]`。每层 cache 为 `(K, V)`，
张量形状均为 `[B, n_heads, cached_length, head_dim]`；缓存的 K 已完成旋转。
传入 cache 时，`ids` 只传新增 token，历史长度加新增长度不能超过 `block_size`。
等价性比较请使用 `model.eval()` 关闭 dropout。

`generate(prompt_ids, max_new_tokens, top_k=0, top_p=1.0, temperature=0.0, use_cache=True)` 支持四种采样，
返回包含 prompt 的完整 token 序列；窗口满时用最近 `block_size` 个 token 重建缓存。
默认 temperature=0，保持 greedy 行为。M4 保存权重时使用
`{"config": model.config, "model_state_dict": model.state_dict()}`，
`load_for_eval` 从同目录读取 `tokenizer.json`。

**输入**：`train.txt`
**输出**：`ckpt/best.pt`、困惑度曲线

`train.py`：next-token prediction，AdamW，cosine schedule，gradient clipping，留出 dev split 监控困惑度。

本地实现读取已有的 `data/train.txt`、`data/dev.txt` 和 M1 的 `ckpt/tokenizer.json`。
在 task-2-mini-gpt 目录中启动唐诗训练：

```bash
python train.py --device cuda --epochs 100 --steps-per-epoch 100 --batch-size 32 --block-size 128 --d-model 128 --n-heads 4 --n-layers 4 --lr 3e-4 --patience 10
python eval/run.py
```

CPU 环境将 `--device cuda` 改为 `--device cpu`。每个 epoch 指 100 个随机采样的
batch（有放回采样），不是遍历语料一遍。训练输入与标签错开一个 token，训练时不使用
KV cache；验证按上下文长度切窗，覆盖完整 dev 集并按 token 数加权计算 loss。
学习率先 warmup 再 cosine 衰减，验证 loss 连续 10 轮未改善会提前停止。

输出包括 `ckpt/best.pt`、`history.json`、`metadata.json` 和 `training_curves.png`，
其中 best.pt 保存验证 loss 最低的一轮。开始新训练时会清除输出目录中的旧训练结果及旧自检结果，
直接保存本轮结果；默认目录为 `ckpt`，`eval/run.py` 默认读取 `ckpt/best.pt`。
训练显示全量 dev 困惑度，官方自检最多评估前 4096 个 token，两者可能略有差异。
是否达到唐诗困惑度 < 50 的 M4 标准，需以实际训练及自检结果判断。

本仓库小语料的另一组已做内存训练验证的配置：

```bash
python train.py --device cuda --tokenizer-vocab-size 320 --epochs 15 --steps-per-epoch 100 --batch-size 32 --block-size 128 --d-model 64 --n-heads 4 --n-layers 2 --dropout 0.2 --lr 1e-3 --weight-decay 1.0 --warmup-ratio 0.05 --min-lr-ratio 0.1 --patience 8 --tie-weights --seed 42
python eval/run.py
```

`--tokenizer-vocab-size` 仅使用训练文本重训 BPE，随后将新词表和新模型保存到同一输出目录，
替换旧结果。320 词表包含 256 个字节、3 个特殊 token 和最多 61 条合并规则。
词表变化会改变 token 粒度，因此不能直接用不同词表的 perplexity 比较语言建模能力。
应同时报告词表大小、编码长度及按原文字节数归一化的损失；此配置通过固定阈值，
不等于在原来的 512 词表实验上将困惑度降到了 50 以下。

**常见坑**：

- 唐诗语料小，容易过拟合：困惑度先降后升就该早停
- 没做 gradient clipping，loss 偶发 spike
- block_size 设得比 dev 文本短没关系（自检会按 `block_size` 非重叠分块），但训练/推理的上下文长度要一致

### 第 17-21 天：采样 + 生成 + 写报告（M5）

**输入**：训练好的模型
**输出**：`src/sampling.py`，几段不同策略的生成样例 + 报告文字

实现 greedy / top-k / top-p / temperature，对比同一 prompt 下的生成多样性与连贯度。

实现位于 `src/sampling.py`，`generate.py` 使用现有权重生成，无需重新训练：

```bash
python generate.py --device cuda --prompt "春江花月夜，" --compare --max-new-tokens 120
python generate.py --device cuda --prompt "春江花月夜，" --strategy top-p --top-p 0.9 --temperature 0.8
python -m unittest discover -s tests -p test_sampling.py -v
```

默认只打印。正式保存四种策略样例时，加 `--output ckpt/generation_samples.json`。
对比模式使用相同 prompt 和 seed；temperature 策略不截断候选，top-k/top-p
分别启用一种截断，二者也应用给定的 temperature。模型 API 支持同时使用 top-k 和 top-p，
按 temperature → top-k → top-p → 归一化 → 随机抽样的顺序执行。
`--no-cache` 可关闭缓存；上下文满后重新计算最近窗口，保证与无缓存的窗口解码一致。

脚本屏蔽训练文本未使用的 BOS/EOS/PAD，按 token 数控制生成长度，token 数不等于中文字数。
默认启用 UTF-8 候选约束：禁止非法字节序列，并按剩余 token 预算保证字符完整结束。
`--repetition-penalty 1.1` 对已出现的字符降权，`--no-repeat-ngram-size 4` 阻止新的重复四字符片段。
重复检查以还原后的字符为单位，避免惩罚中文共用的 UTF-8 前缀字节。
`--raw-sampling` 关闭上述约束，可复现原始采样行为；无效输出仍用替换符显示并标记 `valid_utf8=false`。
默认模式不会补写模板句子，也不会用替换符伪装修复。采样功能通过自检不代表文本必然连贯，
应结合实际样例评估重复、语句通顺程度和多样性。

**常见坑**：

- top-p 要先按概率降序累加到阈值再截断，别忘了重新归一化
- `temperature=0` 应退化为 greedy，别让它触发除零
- 生成时如果不接 KV cache，每步重算全序列会很慢——正好印证 M3 的价值

## 实现约定

`eval/run.py` 会自动检测以下接口；接口对上就能跑自检：

| 文件 | 必须导出 |
|---|---|
| `src/tokenizer.py` | `class BPETokenizer` 含 `encode(text) -> List[int]`、`decode(ids) -> str`、`vocab_size`、`from_pretrained(path)` |
| `src/model.py` | `class MiniGPT(nn.Module)` 含 `forward(ids, kv_cache=None, return_cache=False)`、`generate(prompt_ids, max_new_tokens, top_k, top_p, temperature)`、属性 `block_size`/`max_seq_len`（自检按它对困惑度切窗，取不到默认 256）；`load_for_eval(ckpt_path) -> (model, tokenizer)` |
| `ckpt/tokenizer.json` | 训练好的 BPE 词表 |
| `ckpt/best.pt` | 训练好的模型 state_dict |

接口可以改，但改了请同步调整 `eval/run.py`。

## 自检

```bash
python eval/run.py
```

| 测试 | 通过标准 | 对应 DoD |
|---|---|---|
| `tokenizer_roundtrip` | encode → decode 还原中文文本（除已知 UTF-8 边界 case） | M1 |
| `kv_cache_equivalence` | 开 KV cache 与不开的 logits 一致（误差 < 1e-4） | M3 |
| `perplexity_on_dev` | dev set 困惑度低于阈值（唐诗默认 < 50，TinyStories 默认 < 10） | M4 |

> `perplexity_on_dev` 读取 `data/dev.txt`（最多取前 4096 个词元），按模型上下文长度非重叠分块累加 NLL 后求困惑度——窗口取 `MiniGPT.block_size` 或 `max_seq_len`，取不到则默认 256。因此小上下文模型也能跑通、不会越界或 OOM；建议给 `MiniGPT` 暴露 `block_size` 或 `max_seq_len` 属性，让自检按你的真实训练长度切窗。

结果写入 `eval/result.json`，提交时附上。

## AI Tutor 反馈

把 [eval/tutor_prompt.md](eval/tutor_prompt.md) 整段贴给 Claude / Qwen / DeepSeek，连同你的代码。模型会按统一格式（必检 / 加分 / 优先级）给你针对性 review。

## 前置阅读（非必需）

- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [TinyStories 原论文](https://arxiv.org/abs/2305.07759)
- [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864)
- NNDL2 第 8 章「现代 Transformer 的常见优化」
- 实践书 v2《大语言模型与智能体》「nanoGPT 模型」「预训练循环」「解码 / 采样策略」三节

## 提交

到 [nndl-discussion](https://github.com/nndl/nndl-discussion/discussions) 「llm-beginner 实践成果」分类发帖，附：

1. 你的 fork 仓库链接
2. `eval/result.json` 内容（贴文本即可）
3. DoD checklist 勾选状态
4. 几段生成样例（不同采样策略对比）
5. 200-500 字实验观察：你做了哪些消融、看到了什么有意思的现象（如涌现、外推差异）

## 时间

约 3 周。如果在 M4（训练）卡住，先把模型缩小（如 4 层、d_model=128）在唐诗上跑通整条 pipeline，再扩大规模。
