# 实验二：迁移学习与目标检测

项目只有两个 Python 文件，互相独立。打开文件后按“参数 → 数据 → 模型 → 实验 → 结果”的顺序阅读即可，不需要配置文件或命令行参数。

| 文件 | 实验内容 | 运行方式 |
| --- | --- | --- |
| `lab.py` | 冻结 ResNet18 主干，在 CIFAR-10 上训练分类层 | `python lab.py` |
| `detection.py` | 用预训练 YOLO 检测图片，比较置信度阈值 | `python detection.py` |

## 运行环境

直接使用 NVIDIA 容器自带的 Python，无需安装 CUDA、PyTorch 或新建虚拟环境。当前已验证的环境是 `/opt/conda/bin/python`：PyTorch 2.4.0、torchvision 0.19.0、CUDA 12.1，GPU 为 NVIDIA RTX A6000。

在 VS Code 执行 **Python: Select Interpreter**，选择 `/opt/conda/bin/python`。项目已配置默认解释器；如果之前选过 `.venv`，请手动切换一次。PyCharm 同样选择此已有解释器。

在终端直接运行：

```bash
/opt/conda/bin/python lab.py
/opt/conda/bin/python detection.py
```

检查 GPU：

```bash
/opt/conda/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

当前项目的 `.venv` 是另一套环境，之前安装了 CPU 版 PyTorch，请使用上述容器解释器。`requirements.txt` 不直接列出 torch / torchvision。YOLO 使用 Ultralytics，首次运行前安装辅助依赖；若 pip 提示升级或替换容器的 torch / torchvision，请先检查依赖兼容性：

```bash
/opt/conda/bin/python -m pip install -r requirements.txt
```

若容器运行时出现 OpenCV 缺少 `libxcb.so.1` 或 `libGL.so.1` 的错误，在容器终端安装系统库：`apt-get update && apt-get install -y libxcb1 libgl1 libglib2.0-0`。当前环境已补齐这些库，并使用 NVIDIA RTX A6000 跑通 YOLOv8n 检测。

训练数据需提前准备；首次加载预训练权重时需要联网，之后使用本地缓存。现有 `data/` 和 `.cache/` 可继续使用。

## 实验一：迁移学习

提前将官方 CIFAR-10 Python 压缩包 `cifar-10-python.tar.gz`（约 170 MB）放在 `data/classification/`，或将解压后的 `cifar-10-batches-py/` 文件夹放在该目录。路径可修改 `lab.py` 顶部的 `DATA_DIR`。

运行 `python lab.py`。优先读取已解压数据；只有本地压缩包时，校验完整性后解压。已有目录不会覆盖，数据缺失或损坏时提示手动检查，不会自动下载数据。ResNet18 预训练权重（约 45 MB）仍在首次使用时下载到 `.cache/torch/`。

`assets/detection/` 存放图片准备说明，`data/detection/` 存放实际检测图片，与分类数据分开。

默认从官方训练集中固定随机抽取 1000 张用于训练、200 张用于验证，从官方测试集中抽取 500 张用于最终评价。图片使用预训练权重对应的缩放、裁剪和归一化。默认使用第一张 NVIDIA GPU（`cuda:0`）。

阅读并理解这三个关键步骤：

1. 用 `requires_grad = False` 冻结预训练模型原有参数。
2. 将最后的全连接层替换为输出 10 个类别的新层，仅优化该层参数。
3. 根据验证集准确率保存最佳模型，训练结束后加载它评价测试集。

冻结主干时保持 `model.eval()`，以固定 BatchNorm 统计量。输出为 `[批次大小, 10]` 的类别分数，交叉熵损失直接接收这些分数和整数标签。

参数集中在文件顶部。`DEVICE="cuda:0"` 使用第一张显卡，改为 `"cuda:1"` 使用第二张，改为 `"cpu"` 使用 CPU。分类批次默认为 64，数据加载使用 2 个进程及锁页内存；显存不足时减小 `BATCH_SIZE`。先用验证集比较学习率或轮数，确定参数后再记录最终测试结果，不根据测试结果反复调参。

结果保存在 `outputs/classification/`：`history.csv` 为每轮损失和准确率，`curves.png` 为曲线，`best.pt` 为最佳权重，`test_result.txt` 为测试结果。再次运行会覆盖这些同名结果，需要比较时先复制保存。

思考：为什么要替换分类层？冻结参数有什么作用？训练与验证准确率的差距说明什么？

测试后还会保存 `outputs/classification/misclassified.png`，展示测试顺序中前 9 个误分类样本的原图、真实类别、预测类别和预测分数，方便分析易混淆的类别。可通过 `lab.py` 顶部的 `MISCLASSIFIED_COUNT` 调整展示数量；没有误分类时图片会显示提示。

## 实验二：目标检测

在 `data/detection/` 放入 5—10 张自己拍摄或获准使用的图片，运行 `python detection.py`。图片可包含人、猫、狗、车辆、椅子等 COCO 类别。基础实验与授课内容一致，使用 COCO 预训练 YOLOv8n，首次运行自动下载权重到 `weights/yolov8n.pt`；也可提前放入该权重文件。此实验使用已有模型推理，无需训练。

模型预热后对每张图推理一次，再分别用 0.2、0.5、0.8 过滤同一组经过 NMS 的预测。修改文件顶部的 `THRESHOLDS` 可以比较其他阈值；比较时固定模型、输入尺寸 `IMAGE_SIZE=640` 和 NMS 的 IoU 阈值 `NMS_IOU=0.7`。输入按比例缩放并填充，返回原图坐标下的框。接口说明见 [Ultralytics 官方推理文档](https://docs.ultralytics.com/modes/predict/)。

结果保存在 `outputs/detection/`：每张图片的并排对比图和 `results.csv`。图中包含框、类别和置信度，CSV 记录各阈值框数及该图片共享的一次推理耗时。耗时使用 Ultralytics 返回的模型前向与后处理时间之和，不包含读图、预处理或绘图；仅供观察，不作为严格性能基准。再次运行会覆盖同名结果。

思考：阈值增大时框数如何变化？是否出现误检或漏检？遮挡、小目标和光照是否影响结果？置信度是模型分数；没有标准标注时不计算 mAP，也不把框数当作准确率。

### YOLO 扩展考察（选做）

完成基础实验后，从以下方向至少选择一项。每次只改变一个变量，使用同一组图片，并保存不同设置的结果，避免再次运行时覆盖。

- **NMS 阈值**：固定置信度阈值为 0.5，设置 `THRESHOLDS=[0.5]`，分别运行 `NMS_IOU=0.3、0.5、0.7`，观察密集或遮挡场景中的重复框与漏检，解释 NMS 阈值和置信度阈值的区别。
- **输入尺寸**：固定模型和两种阈值，分别运行 `IMAGE_SIZE=320、640、960`，观察小目标检测和推理耗时的变化。
- **模型规模**：将 `MODEL_PATH` 分别设为 `ROOT / "weights/yolov8n.pt"` 和 `ROOT / "weights/yolov8s.pt"`，固定其他参数，比较检测表现、权重大小和推理耗时。
- **场景鲁棒性**：固定参数，对同一场景改变光照、遮挡或拍摄距离，展示具体失败案例并分析原因。

报告记录模型权重名、输入尺寸、两种阈值、设备及对比结果。比较耗时时应多次运行并报告均值，说明预热和计时范围；无标准标注时仅做定性分析。扩展不要求更换检测网络，也不要求重新训练。

## 提交

提交两个 Python 文件及一份实验报告，报告包括分类曲线、最终测试准确率、至少三张检测对比图和自己的分析。说明使用的数据来源及 AI 辅助范围。无需提交虚拟环境、下载的数据或模型缓存。
