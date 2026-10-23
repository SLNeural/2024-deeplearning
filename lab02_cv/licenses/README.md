# 许可原文与上游源码

| 文件 | 来源与作用 |
|---|---|
| `torchvision-0.19.0-BSD-3-Clause.txt` | 来自 [torchvision v0.19.0/LICENSE](https://raw.githubusercontent.com/pytorch/vision/v0.19.0/LICENSE)，对应软件代码许可，不能据此替预训练权重授予额外权利。 |
| `ultralytics-AGPL-3.0.txt` | 原样复制自当前安装的 Ultralytics 8.4.177 发行包中的 `licenses/LICENSE`。 |
| `ultralytics-8.4.177.tar.gz` | 来自 [PyPI 对应版本源码发行包](https://pypi.org/project/ultralytics/8.4.177/#files)，未修改、未执行，下载后与 PyPI 发布的 SHA-256 比较一致。 |
| `checksums.sha256` | 当前两个权重、许可原文及随附源码包的 SHA-256，用于核对传输完整性，不是额外的授权证明。 |

许可原文应完整保留，包括版权声明、许可条件和免责声明。上游源码包可直接解压阅读，不需要为运行实验而安装它。其版本对应当前教师环境，不代表原始权重的训练版本；版本区别和分发要求见 [模型来源与许可说明](../模型来源与许可说明.md)。

在 `lab02_cv/` 下可执行 `sha256sum -c licenses/checksums.sha256` 核对文件。此命令只检查本地文件是否与记录一致，不代替官方来源及许可核验。
