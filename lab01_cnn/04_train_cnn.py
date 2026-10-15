from pathlib import Path
from datetime import datetime
import csv
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 一、实验配置
SEED = 42
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.01

BASE_DIR = Path(__file__).resolve().parent
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

# 每次运行创建独立目录，避免覆盖之前的实验结果
run_name = (
    f"lr_{LEARNING_RATE}_"
    + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
RUN_DIR = BASE_DIR / "outputs" / run_name
RUN_DIR.mkdir(parents=True, exist_ok=True)

# 固定随机种子，减少对比实验中的随机差异
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_num_threads(2)


# 二、读取已有数据和固定划分
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(
    root=str(BASE_DIR / "data"),
    train=True,
    download=False,
    transform=transform
)

with open(
    BASE_DIR / "outputs" / "split_indices.json",
    encoding="utf-8"
) as f:
    split = json.load(f)

train_data = Subset(dataset, split["train_indices"])
val_data = Subset(dataset, split["val_indices"])

assert set(split["train_indices"]).isdisjoint(
    split["val_indices"]
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=torch.Generator().manual_seed(SEED)
)

# 单独用于每轮结束后评价训练集，读取顺序固定
train_eval_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# 三、小型 CNN，与前一个实验保持一致
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = SmallCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)


# 四、评价函数：只计算指标，不更新参数
def evaluate(loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            count = labels.size(0)
            total_loss += loss.item() * count
            total_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total_samples += count

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


# 五、保存配置及本次使用的数据划分
config = {
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "optimizer": "Adam",
    "normalization_mean": 0.5,
    "normalization_std": 0.5,
    "train_size": len(train_data),
    "val_size": len(val_data),
    "device": str(device),
    "torch_version": str(torch.__version__),
    "selection_rule": "最高验证准确率；相同时保留较早轮次"
}

with open(RUN_DIR / "config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

with open(
    RUN_DIR / "split_indices.json", "w", encoding="utf-8"
) as f:
    json.dump(split, f, indent=2)

print("设备：", device)
print("结果保存目录：", RUN_DIR)
print("开始训练……", flush=True)


# 六、训练，每轮结束后评价训练集和验证集
history = []
best_val_acc = -1.0
best_epoch = 0

for epoch in range(1, EPOCHS + 1):
    model.train()

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 两个集合都使用本轮结束时的同一个模型评价
    train_loss, train_acc = evaluate(train_eval_loader)
    val_loss, val_acc = evaluate(val_loader)

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc
    })

    print(
        f"第 {epoch}/{EPOCHS} 轮 | "
        f"训练 Loss={train_loss:.4f}, Acc={train_acc:.2%} | "
        f"验证 Loss={val_loss:.4f}, Acc={val_acc:.2%}",
        flush=True
    )

    # 只依据验证集选择模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(
            model.state_dict(), RUN_DIR / "best_model.pth"
        )

    # 每轮更新日志，保留所有轮次的结果
    with open(
        RUN_DIR / "history.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(history[0].keys())
        )
        writer.writeheader()
        writer.writerows(history)

torch.save(model.state_dict(), RUN_DIR / "last_model.pth")

with open(RUN_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_acc
        },
        f,
        indent=2
    )


# 七、绘制训练与验证曲线
epochs = [row["epoch"] for row in history]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(
    epochs, [r["train_loss"] for r in history],
    marker="o", label="Train"
)
axes[0].plot(
    epochs, [r["val_loss"] for r in history],
    marker="o", label="Validation"
)
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-entropy")
axes[0].legend()

axes[1].plot(
    epochs, [r["train_accuracy"] for r in history],
    marker="o", label="Train"
)
axes[1].plot(
    epochs, [r["val_accuracy"] for r in history],
    marker="o", label="Validation"
)
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1)
axes[1].legend()

for ax in axes:
    ax.set_xticks(epochs)

fig.tight_layout()
fig.savefig(RUN_DIR / "training_curves.png", dpi=150)
plt.close(fig)

print("\n训练完成！")
print("最佳轮次：", best_epoch)
print(f"最佳验证准确率：{best_val_acc:.2%}")
print("请打开结果目录中的 training_curves.png。")