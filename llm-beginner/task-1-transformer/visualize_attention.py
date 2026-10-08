import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch

from src.model import load_for_eval

ROOT = Path(__file__).resolve().parent


def select_samples(frame, max_length):
    # 按数据原顺序选取，不根据预测正确性或热图形状挑选结果。
    lengths = frame.text.str.len()
    specs = [("positive", "正面样本", (frame.label == 1) & lengths.between(15, 40)),
             ("negative", "负面样本", (frame.label == 0) & lengths.between(15, 40)),
             ("long", "长句样本", lengths.between(50, min(80, max_length)))]
    samples = []
    for slug, title, condition in specs:
        candidates = frame[condition]
        if candidates.empty:
            raise ValueError(f"没有满足长度条件的{title}，请检查测试集或模型长度配置")
        row = candidates.iloc[0]
        samples.append((slug, title, int(candidates.index[0]), str(row.text), int(row.label)))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=ROOT / "ckpt/best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data/test.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--layer", type=int, default=-1, help="从 0 计数；-1 表示最后一层")
    parser.add_argument("--head", type=int, default=0, help="从 0 计数")
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    args = parser.parse_args()
    if not args.font.is_file():
        parser.error("请用 --font 指定支持中文的字体文件")
    model, tokenizer = load_for_eval(str(args.ckpt))
    layer = args.layer if args.layer >= 0 else len(model.blocks) + args.layer
    if not 0 <= layer < len(model.blocks) or not 0 <= args.head < model.blocks[layer].attention.n_heads:
        parser.error("layer 或 head 超出模型范围")
    samples = select_samples(pd.read_parquet(args.data), tokenizer.max_length)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    # matplotlib 字体缓存属于临时产物，退出后自动清理。
    old_config = os.environ.get("MPLCONFIGDIR")
    with tempfile.TemporaryDirectory(prefix="m5-mpl-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.font_manager import FontProperties
            font = FontProperties(fname=str(args.font))
            for slug, title, row, text, label in samples:
                ids = tokenizer(text)
                with torch.no_grad():
                    logits, maps = model(ids[None], return_weights=True)
                    probs = logits.softmax(-1)[0]
                matrix = maps[layer][0, args.head].cpu().numpy()
                assert matrix.shape == (len(ids), len(ids))
                assert np.isfinite(matrix).all()
                np.testing.assert_allclose(matrix.sum(-1), 1, atol=1e-6)
                # 保存所有层和头，方便离线复查；图片只展示指定层和头。
                np.savez_compressed(output / f"{slug}_weights.npz",
                                    **{f"layer_{i}": weights[0].numpy() for i, weights in enumerate(maps)})
                characters = list(text[:len(ids)])
                labels = [f"{i + 1}:{'换行' if c == chr(10) else '空白' if c.isspace() else c}"
                          for i, c in enumerate(characters)]
                size = max(8, len(ids) * 0.20 + 3)
                fig, ax = plt.subplots(figsize=(size, size))
                heatmap = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks(range(len(ids)), labels, rotation=90, fontproperties=font, fontsize=8)
                ax.set_yticks(range(len(ids)), labels, fontproperties=font, fontsize=8)
                ax.set_xlabel("被关注的位置（Key，列）", fontproperties=font)
                ax.set_ylabel("发起关注的位置（Query，行）", fontproperties=font)
                prediction = int(probs.argmax())
                names = ["负面", "正面"]
                ax.set_title(f"{title} · 第 {layer + 1} 层 / 第 {args.head + 1} 头\n"
                             f"真实：{names[label]} ｜ 预测：{names[prediction]} ｜ "
                             f"P(正面)={probs[1]:.3f} ｜ {len(ids)} 字符", fontproperties=font)
                fig.colorbar(heatmap, ax=ax, fraction=0.035, pad=0.025).set_label("Attention weight")
                fig.tight_layout()
                fig.savefig(output / f"{slug}_attention.png", dpi=160)
                plt.close(fig)
                received = matrix.mean(axis=0)
                top = np.argsort(-received, kind="stable")[:5]
                results.append(dict(name=slug, title=title, dataset_row=row, text=text,
                                    label=label, prediction=prediction, positive_probability=float(probs[1]),
                                    tokens=len(ids), unknown_count=int(ids.eq(1).sum()),
                                    layer_index=layer, head_index=args.head,
                                    top_received=[dict(position=int(i)+1, char=characters[i],
                                                       mean_weight=float(received[i])) for i in top]))
                print(f"{slug}: label={label}, prediction={prediction}, P(positive)={probs[1]:.4f}", flush=True)
        finally:
            if old_config is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = old_config
    record = dict(checkpoint=str(args.ckpt.resolve()),
                  checkpoint_sha256=hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
                  data=str(args.data.resolve()), selection="原始顺序首条：正/负面15–40字符，长句50–80字符",
                  samples=results)
    (output / "samples.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    paragraphs = ["# M5 注意力观察\n",
                  f"使用 M3 最佳模型，在测试集中按固定长度和标签规则选取三条样本，"
                  f"未按预测结果筛选。统一展示第 {layer+1} 层、第 {args.head+1} 头，色标固定为 0–1。"
                  "横轴是 Key，纵轴是 Query，序号从 1 开始；每一行权重之和为 1。\n"]
    for result in results:
        top_text = "、".join(f"第{x['position']}位{repr(x['char'])}（{x['mean_weight']:.3f}）"
                            for x in result["top_received"][:3])
        paragraphs.append(f"{result['title']}共{result['tokens']}字符，预测{'正确' if result['label']==result['prediction'] else '错误'}。"
                          f"按所有 Query 的平均关注权重，最受关注的位置为{top_text}。\n")
    paragraphs.append("上述统计仅描述一个头的信息聚合，不能据此断言这些字符造成了分类结果。"
                      "分类器使用双向注意力，所以对角线右上方有权重是正常现象，与 M4 的因果模型不同。"
                      "长句每行要向更多位置分配权重，颜色偏浅不等于模型更不确定；"
                      "完整判断还需比较其他层和头，并结合遮挡实验。\n")
    paragraphs.append("## 样本与图片\n")
    for result in results:
        paragraphs.append(f"### {result['title']}\n\n{result['text']}\n\n"
                          f"![{result['title']}]({result['name']}_attention.png)\n")
    (output / "observations.md").write_text("\n".join(paragraphs), encoding="utf-8")
    print(f"M5 产物：{output}", flush=True)


if __name__ == "__main__":
    main()
