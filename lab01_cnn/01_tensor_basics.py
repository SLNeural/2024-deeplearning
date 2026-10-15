import torch

# 一、创建 Tensor（张量）
print("\n===== 1. 创建张量 =====")

x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

print("张量内容：")
print(x)
print("形状：", x.shape)
print("维度数量：", x.ndim)
print("数据类型：", x.dtype)


# 二、reshape：改变形状，元素总数不变
print("\n===== 2. 改变张量形状 =====")

y = x.reshape(3, 2)

print("原来的形状：", x.shape)
print("改变后的形状：", y.shape)
print("改变后的内容：")
print(y)


# 三、广播：给每一行加上同一个向量
print("\n===== 3. 广播运算 =====")

b = torch.tensor([10.0, 20.0, 30.0])
result = x + b

print("x 的形状：", x.shape)
print("b 的形状：", b.shape)
print("x + b 的结果：")
print(result)


# 四、自动求导
print("\n===== 4. 自动求导 =====")

# requires_grad=True 表示需要计算关于 w 的梯度
w = torch.tensor(3.0, requires_grad=True)

# 定义函数 f(w) = w² + 2w
f = w ** 2 + 2 * w

# 反向传播，计算梯度
f.backward()

print("w 的值：", w.item())
print("f(w) 的值：", f.item())
print("自动计算的梯度：", w.grad.item())
print("手算结果：2 × 3 + 2 = 8")


# 五、模拟一批灰度图像
print("\n===== 5. 图像批量张量 =====")

# 这里是随机数据，用于理解形状，不是真实 MNIST 图片
# 顺序：批量大小、通道数、高度、宽度
images = torch.randn(8, 1, 28, 28)

print("一批图像的形状：", images.shape)
print("含义：8 张图像，每张 1 个通道，高 28、宽 28")
print("第一张图像的形状：", images[0].shape)


# 六、选择计算设备：有 GPU 就使用 GPU，否则使用 CPU
print("\n===== 6. 计算设备 =====")

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

images = images.to(device)

print("当前设备：", device)
print("图像张量所在设备：", images.device)

print("\n实验运行结束！")