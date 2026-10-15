"""实验一：冻结 ResNet18 主干，在 CIFAR-10 上训练新的分类层。"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import check_integrity, extract_archive
from torchvision.models import ResNet18_Weights, resnet18

# 学生主要修改这里；先根据验证集调整参数，最后查看测试结果。
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data/classification"
OUTPUT_DIR = ROOT / "outputs/classification"
TRAIN_SIZE = 1000
VAL_SIZE = 200
TEST_SIZE = 500
BATCH_SIZE = 64
NUM_WORKERS = 2
DEVICE = "cuda:0"  # 第二张显卡改为 cuda:1；使用 CPU 改为 cpu
EPOCHS = 5
LEARNING_RATE = 0.001
SEED = 42
MISCLASSIFIED_COUNT = 9  # 最多展示几个测试集误分类样本


def load_local_data(transform):
    """只使用本地 CIFAR-10；已有目录不覆盖，没有目录时可解压官方包。"""
    folder = DATA_DIR / "cifar-10-batches-py"
    archive = DATA_DIR / "cifar-10-python.tar.gz"
    if not folder.exists():
        if not archive.is_file():
            raise FileNotFoundError(
                f"请提前把 cifar-10-python.tar.gz 或解压后的 cifar-10-batches-py 放到 {DATA_DIR}。"
                "程序不会自动下载数据。")
        if not check_integrity(str(archive), CIFAR10.tgz_md5):
            raise RuntimeError(f"本地压缩包不完整或不是官方 CIFAR-10 Python 包：{archive}；请检查文件。")
        print(f"解压本地数据：{archive}", flush=True)
        extract_archive(str(archive), str(DATA_DIR))
    try:
        train_data = CIFAR10(DATA_DIR, train=True, download=False, transform=transform)
        test_data = CIFAR10(DATA_DIR, train=False, download=False, transform=transform)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{folder} 数据缺失或校验失败。请检查并手动修复；程序不会下载或覆盖已有数据。") from exc
    print(f"使用本地 CIFAR-10：{folder}", flush=True)
    return train_data, test_data


def run_epoch(model, loader, device, optimizer=None, mistakes=None):
    """传入优化器时训练，否则只评价；返回平均损失和准确率。"""
    # 冻结主干也要固定 BatchNorm 的统计量。
    model.eval()
    model.fc.train(optimizer is not None)
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        if mistakes is not None:
            probabilities = logits.detach().softmax(1)
            scores, predictions = probabilities.max(1)
            for offset in (predictions != labels).nonzero(as_tuple=True)[0].tolist():
                if len(mistakes) >= MISCLASSIFIED_COUNT:
                    break
                mistakes.append((total + offset, int(labels[offset]),
                                 int(predictions[offset]), float(scores[offset])))
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def save_misclassified(test_data, test_indices, mistakes):
    """展示测试顺序中前几个错误，使用未归一化的 CIFAR-10 原图。"""
    columns = 3
    rows = max(1, (len(mistakes) + columns - 1) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(10, 3 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, (position, truth, prediction, score) in zip(axes.flat, mistakes):
        index = test_indices[position]
        axis.imshow(test_data.data[index], interpolation="nearest")
        axis.set_title(f"True: {test_data.classes[truth]}\n"
                       f"Pred: {test_data.classes[prediction]} ({score:.1%})\n"
                       f"Test index: {index}", fontsize=10)
    if not mistakes:
        axes[0, 1].text(0.5, 0.5, "No misclassified samples", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "misclassified.png", dpi=150)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.hub.set_dir(str(ROOT / ".cache/torch"))
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请在 IDE 选择容器自带的 /opt/conda/bin/python，或检查容器 GPU 是否已挂载")
    device = torch.device(DEVICE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"使用设备：{device}", flush=True)
    if device.type == "cuda":
        print(f"显卡：{torch.cuda.get_device_name(device)}", flush=True)

    # 1. 官方训练集随机分为训练/验证子集；测试集始终独立。
    weights = ResNet18_Weights.DEFAULT
    train_data, test_data = load_local_data(weights.transforms())
    if not (TRAIN_SIZE > 0 and VAL_SIZE > 0 and TRAIN_SIZE + VAL_SIZE <= len(train_data)
            and 0 < TEST_SIZE <= len(test_data) and EPOCHS > 0 and BATCH_SIZE > 0):
        raise ValueError("样本数、轮数和批次大小必须为正数，且样本数不能超过数据集容量")
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(train_data), generator=generator).tolist()
    test_indices = torch.randperm(len(test_data), generator=generator)[:TEST_SIZE].tolist()
    train_loader = DataLoader(Subset(train_data, indices[:TRAIN_SIZE]), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda", generator=generator)
    val_loader = DataLoader(Subset(train_data, indices[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]),
                            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")
    test_loader = DataLoader(Subset(test_data, test_indices), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")

    # 2. 使用 ImageNet 预训练权重，冻结原参数，再替换分类层。
    model = resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))
    model.to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    print(f"训练/验证/测试：{TRAIN_SIZE}/{VAL_SIZE}/{TEST_SIZE}")
    print(f"可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 3. 训练分类层，按验证准确率保存最佳模型。
    history, best_accuracy = [], -1.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device)
        history.append([epoch, train_loss, train_acc, val_loss, val_acc])
        print(f"第 {epoch}/{EPOCHS} 轮 | 训练 loss={train_loss:.4f}, acc={train_acc:.3f}"
              f" | 验证 loss={val_loss:.4f}, acc={val_acc:.3f}", flush=True)
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "best.pt")
    with (OUTPUT_DIR / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        writer.writerows(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, train_column, val_column, title in [(axes[0], 1, 3, "Loss"), (axes[1], 2, 4, "Accuracy")]:
        axis.plot([r[0] for r in history], [r[train_column] for r in history], label="train")
        axis.plot([r[0] for r in history], [r[val_column] for r in history], label="validation")
        axis.set(xlabel="Epoch", title=title)
        axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "curves.png")
    plt.close(fig)

    # 4. 恢复验证集选出的模型，只在训练结束后评价测试集。
    model.load_state_dict(torch.load(OUTPUT_DIR / "best.pt", map_location=device, weights_only=True))
    mistakes = []
    test_loss, test_acc = run_epoch(model, test_loader, device, mistakes=mistakes)
    save_misclassified(test_data, test_indices, mistakes)
    result = f"测试 loss={test_loss:.4f}, accuracy={test_acc:.4f}\n"
    (OUTPUT_DIR / "test_result.txt").write_text(result, encoding="utf-8")
    print(result, end="")
    print(f"结果已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
