"""演示第 4 步：点击运行，检查 RNN 结构、张量形状与单批次更新。"""

from train import main


if __name__ == "__main__":
    main(default_model="rnn", check_only=True)
