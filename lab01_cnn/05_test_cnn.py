from pathlib import Path
import csv
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 一、指定根据验证集选定的实验目录
BASE_DIR = Path(__file__).resolve().parent

RUN_DIR = (
    BASE_DIR / "outputs"
    / "lr_0.01_20260929_101544_212054"
)

MODEL_PATH = RUN_DIR / "best_model.pth"
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"没有找到模型，请检查 RUN_DIR：{MODEL_PATH}"
    )

RESULT_DIR = RUN_DIR / "test_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

with open(RUN_DIR / "config.json", encoding="utf-8") as f:
    config = json.load(f)

with open(RUN_DIR / "summary.json", encoding="utf-8") as f:
    summary = json.load(f)


# 二、搭建与训练时完全相同的网络
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

# 加载之前训练得到的参数
state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)
model.load_state_dict(state_dict)
model.eval()

print("加载模型：", MODEL_PATH)
print("所选学习率：", config["learning_rate"])
print("所选轮次：", summary["best_epoch"])
print("计算设备：", device)


# 三、读取官方测试集，预处理与训练时一致
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (config["normalization_mean"],),
        (config["normalization_std"],)
    )
])

test_data = datasets.MNIST(
    root=str(BASE_DIR / "data"),
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

print("测试图片数量：", len(test_data))


# 四、预测整个测试集，不计算梯度、不更新参数
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().tolist())

y_true = np.array(all_labels)
y_pred = np.array(all_predictions)

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=list(range(10)),
    average="macro",
    zero_division=0
)

cm = confusion_matrix(
    y_true, y_pred, labels=list(range(10))
)
wrong_indices = np.flatnonzero(y_true != y_pred)

print(f"\n测试集准确率：{accuracy:.2%}")
print(f"测试集宏平均 F1：{macro_f1:.4f}")
print("误分类数量：", len(wrong_indices))


# 五、保存评价指标
metrics = {
    "selected_run": RUN_DIR.name,
    "learning_rate": config["learning_rate"],
    "selected_epoch": summary["best_epoch"],
    "best_val_accuracy": summary["best_val_accuracy"],
    "test_size": len(test_data),
    "test_accuracy": float(accuracy),
    "test_macro_f1": float(macro_f1),
    "error_count": int(len(wrong_indices))
}

with open(
    RESULT_DIR / "test_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

# 保存全部预测，便于追溯误分类样本
with open(
    RESULT_DIR / "test_predictions.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow(["index", "true_label", "predicted_label"])
    for index, (true, pred) in enumerate(zip(y_true, y_pred)):
        writer.writerow([index, int(true), int(pred)])


# 六、绘制混淆矩阵
fig, ax = plt.subplots(figsize=(8, 7))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=list(range(10))
)
display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)
ax.set_title("MNIST Test Confusion Matrix")
fig.tight_layout()
fig.savefig(RESULT_DIR / "confusion_matrix.png", dpi=150)
plt.close(fig)

np.savetxt(
    RESULT_DIR / "confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d"
)


# 七、展示按测试集顺序出现的前 16 个错误
# 固定展示规则，不挑选最有利的样本
fig, axes = plt.subplots(4, 4, figsize=(9, 9))

for ax in axes.flat:
    ax.axis("off")

for ax, index in zip(axes.flat, wrong_indices[:16]):
    image, label = test_data[int(index)]

    # 撤销归一化，仅用于显示
    image = (
        image[0] * config["normalization_std"]
        + config["normalization_mean"]
    )

    ax.imshow(image.numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(
        f"Index: {index}\n"
        f"True: {label}, Pred: {y_pred[index]}"
    )

if len(wrong_indices) == 0:
    fig.suptitle("No misclassified samples")
else:
    fig.suptitle("First misclassified samples in test order")

fig.tight_layout()
fig.savefig(RESULT_DIR / "misclassified.png", dpi=150)
plt.close(fig)

print("\n测试完成，结果保存到：", RESULT_DIR)
