"""演示第 2 步：点击运行，检查 LSTM 结构、张量形状与单批次更新。"""

from train import main


if __name__ == "__main__":
    main(default_model="lstm", check_only=True)
