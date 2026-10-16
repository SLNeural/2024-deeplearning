"""读取真实训练和最终测试结果，生成比较表、指标柱状图和训练曲线。"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from utils import (OUTPUT_DIR, read_json, write_csv, sha256_file,
                   check_same_conditions)


def compare(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    metrics, histories = [], {}
    for model_type in ("rnn", "lstm"):
        directory = output_dir / model_type
        result = read_json(directory / "test_metrics.json")
        summary = read_json(directory / "summary.json")
        digest = sha256_file(directory / "best.pt")
        if result["checkpoint_sha256"] != digest or summary["checkpoint_sha256"] != digest:
            raise ValueError(f"{model_type} 训练后模型已改变，请先重新运行 evaluate.py")
        if (result["config"] != summary["config"]
                or result["training_seconds"] != summary["training_seconds"]):
            raise ValueError(f"{model_type} 测试结果已过期，请先重新运行 evaluate.py")
        metrics.append(result)
        with (directory / "history.csv").open(encoding="utf-8-sig", newline="") as handle:
            histories[model_type] = list(csv.DictReader(handle))
    check_same_conditions([item["config"] for item in metrics])
    if metrics[0]["test_total"] != metrics[1]["test_total"]:
        raise ValueError("两个测试集大小不同，不能比较")
    if metrics[0].get("test_set_id") != metrics[1].get("test_set_id"):
        raise ValueError("两个测试集的原文或标签不同，不能比较")
    destination = output_dir / "comparison"
    destination.mkdir(parents=True, exist_ok=True)
    fields = ["model", "test_accuracy", "parameters", "training_seconds", "best_epoch", "best_val_accuracy"]
    rows = [{field: (item[field].upper() if field == "model" else item[field]) for field in fields}
            for item in metrics]
    write_csv(destination / "comparison.csv", rows, fields)
    markdown = ["| 模型 | 测试 Accuracy | 参数量 | 累计训练时间（秒） | 最佳验证轮次 |",
                "|---|---:|---:|---:|---:|"]
    for row in rows:
        markdown.append(f"| {row['model']} | {row['test_accuracy']:.4f} | {row['parameters']:,} | "
                        f"{row['training_seconds']:.3f} | {row['best_epoch']} |")
    markdown.extend(["", "时间仅累计各 Epoch 的训练批次，含加载、前向、反向与 Adam 更新；"
                     "不含下载、单批次检查、验证、保存、绘图和测试。",
                     "这是单次固定种子实验，只描述当前数据与设置下的结果，不能证明某模型普遍更优。",
                     f"输入最大长度：{metrics[0]['config']['max_length']}；"
                     f"隐藏表示汇总：{metrics[0]['config'].get('pooling', 'last')}。",
                     "数据 ID：" + metrics[0]["data_id"]])
    (destination / "comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    names = [row["model"] for row in rows]
    for ax, key, title in zip(axes, ("test_accuracy", "parameters", "training_seconds"),
                              ("Test accuracy", "Trainable parameters", "Training time (s)")):
        values = [row[key] for row in rows]
        bars = ax.bar(names, values, color=["#5680b5", "#e99a53"])
        labels = [f"{value:.3f}" if key != "parameters" else f"{value:,}" for value in values]
        ax.bar_label(bars, labels=labels, padding=3)
        ax.set_title(title)
        ax.set_ylim(0, 1.12 if key == "test_accuracy" else max(values) * 1.2 + 0.001)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination / "model_comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for model_type, history in histories.items():
        epochs = [int(row["epoch"]) for row in history]
        axes[0].plot(epochs, [float(row["train_loss"]) for row in history], "o-", label=model_type.upper())
        axes[1].plot(epochs, [float(row["val_accuracy"]) for row in history], "o-", label=model_type.upper())
    axes[0].set(xlabel="Epoch", ylabel="Training loss")
    axes[1].set(xlabel="Epoch", ylabel="Validation accuracy", ylim=(0, 1))
    for ax in axes:
        ax.set_xticks(epochs)
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination / "training_comparison.png", dpi=150)
    plt.close(fig)
    print("\n".join(markdown[:4]))
    print(f"比较表、指标柱状图和训练曲线已保存：{destination}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    try:
        compare(args.output_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"比较失败：{exc}\n请先依次运行 train.py、train_rnn.py 和 evaluate.py。\n")


if __name__ == "__main__":
    main()
