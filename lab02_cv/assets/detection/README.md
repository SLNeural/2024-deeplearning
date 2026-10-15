# 检测图片准备

在 `data/detection/` 放入 5—10 张本人拍摄或获准使用的 JPG、PNG、WebP 等图片，然后运行 `python detection.py`。

选择包含猫、狗、车辆、椅子等 COCO 类别的场景，尝试不同光照、遮挡和目标大小。在实验报告中记录图片来源；可参考同目录的 `sources.example.csv`，程序无需读取该表。

在 `outputs/detection/` 查看并排对比图及 `results.csv`，自行描述明显误检和漏检。没有标准标注时不报告严格 Precision、Recall 或 mAP。

基础实验使用 COCO 预训练 YOLOv8n，对同一张图片比较 0.2、0.5、0.8 三个置信度阈值，固定输入尺寸和 NMS 的 IoU 阈值。

扩展可在 YOLO 基础上选择 NMS 阈值、输入尺寸、模型规模或光照与遮挡实验，每次只改变一个变量。具体参数与报告要求见项目根目录的 `README.md`。不同设置的输出请分别保存，避免覆盖。
