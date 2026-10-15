from pathlib import Path
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# 一、基本设置
BASE_DIR = Path(__file__).resolve().parent
torch.manual_seed(42)

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)
print("计算设备：", device)


# 二、读取已有数据和固定划分
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_train = datasets.MNIST(
    root=str(BASE_DIR / "data"),
    train=True,
    download=False,
    transform=transform
)

with open(
    BASE_DIR / "outputs" / "split_indices.json",
    "r",
    encoding="utf-8"
) as f:
    split = json.load(f)

train_data = Subset(full_train, split["train_indices"])

train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    generator=torch.Generator().manual_seed(42)
)


# 三、搭建小型 CNN
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # 输入：[B, 1, 28, 28]
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 输出：[B, 8, 14, 14]

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
            # 输出：[B, 16, 7, 7]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
            # 输出：[B, 10]
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = SmallCNN().to(device)
print("\n网络结构：")
print(model)

parameter_count = sum(p.numel() for p in model.parameters())
print("参数总数：", parameter_count)


# 四、定义损失函数和优化器
# 交叉熵接收原始类别分数，不要提前加 Softmax
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 五、取一个批次
images, labels = next(iter(train_loader))
images = images.to(device)
labels = labels.to(device)

print("\n输入图片形状：", images.shape)
print("真实标签形状：", labels.shape)

# 保存更新前的第一层卷积权重，用于比较
weight_before = model.features[0].weight.detach().clone()


# 六、完成一次训练更新
model.train()

optimizer.zero_grad()          # 1. 清除旧梯度
outputs = model(images)        # 2. 前向计算
loss = criterion(outputs, labels)  # 3. 计算损失
loss.backward()                # 4. 反向传播，计算梯度

print("网络输出形状：", outputs.shape)
print("更新前的本批次损失：", loss.item())
print(
    "第一层权重的平均绝对梯度：",
    model.features[0].weight.grad.abs().mean().item()
)

optimizer.step()               # 5. 根据梯度更新参数


# 七、确认参数发生变化
weight_after = model.features[0].weight.detach()
change = (weight_after - weight_before).abs().max().item()

print("\n第一层权重的最大绝对变化：", change)
print("第一层权重是否发生变化：", change > 0)


# 八、查看更新后在同一批次上的输出
model.eval()

with torch.no_grad():
    new_outputs = model(images)
    new_loss = criterion(new_outputs, labels)
    predictions = new_outputs.argmax(dim=1)
    batch_accuracy = (predictions == labels).float().mean()

print("更新后的本批次损失：", new_loss.item())
print("前 10 个真实标签：", labels[:10].tolist())
print("前 10 个预测标签：", predictions[:10].tolist())
print(f"更新后的本批次准确率：{batch_accuracy.item():.2%}")

print("\n单批次训练完成！")
print("注意：这里的准确率不是验证集或测试集准确率。")