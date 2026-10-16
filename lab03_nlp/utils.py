"""少量公共函数：路径、随机种子、文件读写与统一评价。"""

import csv
import hashlib
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无图形桌面也能保存图片。
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
LABEL_NAMES = {0: "负面", 1: "正面"}
FAIR_KEYS = ("data_id", "seed", "epochs", "batch_size", "embedding_dim",
             "hidden_dim", "max_length", "learning_rate", "threads", "device",
             "optimizer", "loss", "num_layers", "bidirectional",
             "pooling", "dropout", "weight_decay", "grad_clip")
LEGACY_DEFAULTS = {"pooling": "last", "dropout": 0.0, "weight_decay": 0.0, "grad_clip": 0.0}


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 BOM 便于用 Excel 打开中文；空误分类结果也保留表头。
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    content = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def set_seed(seed=42, threads=2):
    if threads < 1:
        raise ValueError("CPU 线程数必须至少为 1")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate_loader(model, loader):
    """训练时只传验证集；最终测试时才传测试集。按样本数累计指标。"""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    predictions, labels = [], []
    criterion = torch.nn.CrossEntropyLoss()
    for inputs, targets, lengths in loader:
        logits = model(inputs, lengths)
        predicted = logits.argmax(dim=1)
        loss_sum += criterion(logits, targets).item() * targets.size(0)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
        predictions.extend(predicted.tolist())
        labels.extend(targets.tolist())
    if total == 0:
        raise ValueError("评价集为空，请重新准备数据")
    return {"loss": loss_sum / total, "accuracy": correct / total,
            "correct": correct, "total": total,
            "predictions": predictions, "labels": labels}


def check_same_conditions(configs):
    first = configs[0]
    for other in configs[1:]:
        different = [key for key in FAIR_KEYS
                     if first.get(key, LEGACY_DEFAULTS.get(key))
                     != other.get(key, LEGACY_DEFAULTS.get(key))]
        if different:
            raise ValueError("两模型的比较条件不同：" + ", ".join(different)
                             + "。请用相同参数重新训练，测试集不能用于调参。")
