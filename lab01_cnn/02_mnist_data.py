from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")  # 远程服务器无需弹出图形窗口
import matplotlib.pyplot as plt


# 一、设置路径
# 以当前代码文件的位置为准，点击三角按钮运行也不会保存错位置
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32


# 二、图像预处理
transform = transforms.Compose([
    # 将图片转换为 Tensor，并将像素从 0~255 缩放到 0~1
    transforms.ToTensor(),

    # 使用固定的均值、标准差参数，将 0~1 映射到 -1~1
    # 这里的 0.5 是预设值，不是从测试集计算得到的
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])


# 三、读取 MNIST 官方训练集
# 首次运行会下载；已有完整数据时不重复下载
full_train = datasets.MNIST(
    root=str(DATA_DIR),
    train=True,
    download=True,
    transform=transform
)

print("官方训练集图片数量：", len(full_train))


# 四、固定训练子集和验证集
split_path = OUTPUT_DIR / "split_indices.json"

if split_path.exists():
    # 后续运行直接读取同一份划分
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
else:
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(
        len(full_train), generator=generator
    ).tolist()

    split = {
        "seed": SEED,
        "train_indices": indices[:6000],
        "val_indices": indices[6000:7000]
    }

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

train_data = Subset(full_train, split["train_indices"])
val_data = Subset(full_train, split["val_indices"])

# 检查训练样本与验证样本没有重叠
assert set(split["train_indices"]).isdisjoint(
    split["val_indices"]
)

print("本实验训练图片数量：", len(train_data))
print("本实验验证图片数量：", len(val_data))
print("其他训练图片暂不使用；官方测试集留到最终评价。")


# 五、用 DataLoader 分批读取
train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=torch.Generator().manual_seed(SEED)
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# 取出一个批次
images, labels = next(iter(train_loader))

print("\n图片批量形状：", images.shape)
print("标签批量形状：", labels.shape)
print("图片数据类型：", images.dtype)
print("标签数据类型：", labels.dtype)
print("前 16 个标签：", labels[:16].tolist())
print("本批次像素最小值：", images.min().item())
print("本批次像素最大值：", images.max().item())

assert images.shape[1:] == (1, 28, 28)
assert labels.min().item() >= 0
assert labels.max().item() <= 9


# 六、保存前 16 张图片，核对图片与标签
fig, axes = plt.subplots(4, 4, figsize=(7, 7))

for i, ax in enumerate(axes.flat):
    # 撤销归一化，恢复到 0~1，用于显示
    image = images[i, 0] * 0.5 + 0.5

    ax.imshow(image.numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Label: {labels[i].item()}")
    ax.axis("off")

fig.tight_layout()

figure_path = OUTPUT_DIR / "mnist_samples.png"
fig.savefig(figure_path, dpi=150)
plt.close(fig)

print("\n样本图片已保存：", figure_path)
print("数据划分已保存：", split_path)
print("数据处理实验完成！")