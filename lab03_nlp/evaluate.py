"""两模型训练结束后，加载最佳验证模型，在同一测试集上最终评价。"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models import model_from_config
from prepare_data import SentimentDataset, load_prepared
from utils import (DATA_DIR, OUTPUT_DIR, LABEL_NAMES, set_seed, read_json, write_json,
                   write_csv, evaluate_loader, sha256_file, check_same_conditions, object_hash)


def load_best(directory, vocab, data_id):
    checkpoint_path = directory / "best.pt"
    summary = read_json(directory / "summary.json")
    digest = sha256_file(checkpoint_path)
    if summary["checkpoint_sha256"] != digest:
        raise ValueError(f"{directory.name} 模型与训练摘要不一致，请重新完成训练")
    # 仅加载本项目保存的张量及基本类型，兼容较新的 PyTorch 默认安全加载。
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["data_id"] != data_id or checkpoint["vocab"] != vocab:
        raise ValueError(f"{directory.name} 使用了不同的数据/字符表，请重新训练")
    if (checkpoint["config"] != summary["config"]
            or checkpoint["best_epoch"] != summary["best_epoch"]
            or checkpoint["best_val_accuracy"] != summary["best_val_accuracy"]):
        raise ValueError(f"{directory.name} 最佳轮次、配置或验证指标与摘要不一致")
    model = model_from_config(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model, summary


def evaluate(data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    vocab, manifest = load_prepared(data_dir)
    # 先确认两模型均训练完且比较条件一致，然后才进行测试。
    models, summaries = {}, {}
    for model_type in ("lstm", "rnn"):
        models[model_type], summaries[model_type] = load_best(
            output_dir / model_type, vocab, manifest["data_id"])
    check_same_conditions([summary["config"] for summary in summaries.values()])
    config = summaries["lstm"]["config"]
    set_seed(config["seed"], config["threads"])
    rows = read_json(data_dir / "test.json")
    # 与输入长度无关，便于核对优化前后确实评价了同一批原文及标签。
    test_set_id = object_hash([{key: row[key] for key in ("id", "text", "label")} for row in rows])
    dataset = SentimentDataset(rows, vocab, manifest["max_length"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    errors, seen_error_texts, results = [], set(), {}
    fields = ["model", "id", "text", "visible_text", "effective_length", "truncated",
              "true_label", "predicted_label", "true_sentiment", "predicted_sentiment"]
    for model_type, model in models.items():
        result = evaluate_loader(model, loader)
        predictions = []
        for row, predicted in zip(rows, result["predictions"]):
            record = {"model": model_type.upper(), "id": row["id"], "text": row["text"],
                      "visible_text": row["text"][:manifest["max_length"]],
                      "effective_length": min(len(row["text"]), manifest["max_length"]),
                      "truncated": len(row["text"]) > manifest["max_length"],
                      "true_label": row["label"], "predicted_label": predicted,
                      "true_sentiment": LABEL_NAMES[row["label"]],
                      "predicted_sentiment": LABEL_NAMES[predicted]}
            predictions.append(record)
            # 总共至多三条，按固定测试顺序选取；先 LSTM，不足时用 RNN 补足。
            if (predicted != row["label"] and len(errors) < 3
                    and row["text"] not in seen_error_texts):
                errors.append(record)
                seen_error_texts.add(row["text"])
        summary = summaries[model_type]
        metrics = {"model": model_type, "test_accuracy": result["accuracy"],
                   "test_loss": result["loss"], "test_correct": result["correct"],
                   "test_total": result["total"], "parameters": summary["parameters"],
                   "training_seconds": summary["training_seconds"],
                   "best_epoch": summary["best_epoch"],
                   "best_val_accuracy": summary["best_val_accuracy"],
                   "config": summary["config"], "data_id": manifest["data_id"],
                   "test_set_id": test_set_id,
                   "checkpoint_sha256": summary["checkpoint_sha256"]}
        destination = output_dir / model_type
        write_json(destination / "test_metrics.json", metrics)
        write_csv(destination / "test_predictions.csv", predictions, fields)
        print(f"{model_type.upper()}: 最佳验证 Epoch={summary['best_epoch']}，"
              f"测试 Accuracy={result['accuracy']:.4f} ({result['correct']}/{result['total']})")
        results[model_type] = metrics
    write_csv(output_dir / "misclassified.csv", errors, fields)
    for index, row in enumerate(errors, 1):
        print(f"\n误分类 {index} [{row['model']}] 真实={row['true_sentiment']}，"
              f"预测={row['predicted_sentiment']}，截断={row['truncated']}")
        print("原始文本：", row["text"])
        print("模型实际可见的文本：", row["visible_text"])
    print(f"\n共导出 {len(errors)} 条不重复误分类至 {output_dir / 'misclassified.csv'}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    try:
        evaluate(args.data_dir, args.output_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"测试失败：{exc}\n请确认两个模型均已完成训练，且使用相同数据与参数。\n")


if __name__ == "__main__":
    main()
