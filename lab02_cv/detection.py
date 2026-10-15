"""实验二：使用预训练 YOLO 检测图片，比较不同置信度阈值。"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "data/detection"  # 在这里放入自己的图片
OUTPUT_DIR = ROOT / "outputs/detection"
THRESHOLDS = [0.2, 0.5, 0.8]
MODEL_PATH = ROOT / "weights/yolov8n.pt"
IMAGE_SIZE = 640
NMS_IOU = 0.7
DEVICE = "cuda:0"  # 第二张显卡改为 cuda:1；使用 CPU 改为 cpu


def main():
    # 1. 读取图片，加载 COCO 预训练模型；无需训练或标注。
    paths = sorted(p for p in IMAGE_DIR.glob("*")
                   if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
    if not paths:
        raise FileNotFoundError(f"请先在 {IMAGE_DIR} 放入 JPG、PNG 等图片")
    if not THRESHOLDS or any(not 0 <= t <= 1 for t in THRESHOLDS):
        raise ValueError("置信度阈值必须在 0 到 1 之间")
    torch.set_num_threads(4)
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请在 IDE 选择容器自带的 /opt/conda/bin/python，或检查容器 GPU 是否已挂载")
    device = torch.device(DEVICE)
    if device.type == "cuda":
        print(f"显卡：{torch.cuda.get_device_name(device)}", flush=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    categories = model.names
    predict_options = dict(conf=min(THRESHOLDS), iou=NMS_IOU, imgsz=IMAGE_SIZE,
                           device=DEVICE, verbose=False, save=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"使用设备：{device}；共 {len(paths)} 张图片", flush=True)

    for index, path in enumerate(paths, 1):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        # 2. 推理一次，再用不同阈值过滤同一组预测，方便公平比较。
        with torch.inference_mode():
            model.predict(image, **predict_options)
            result = model.predict(image, **predict_options)[0]
        elapsed_ms = result.speed["inference"] + result.speed["postprocess"]
        prediction = {"boxes": result.boxes.xyxy.cpu(),
                      "labels": result.boxes.cls.cpu(),
                      "scores": result.boxes.conf.cpu()}

        # 3. 画出框、类别和分数，保存并排对比图及框数表。
        fig, axes = plt.subplots(1, len(THRESHOLDS), figsize=(6 * len(THRESHOLDS), 5), squeeze=False)
        for axis, threshold in zip(axes[0], THRESHOLDS):
            axis.imshow(image)
            keep = prediction["scores"] >= threshold
            for box, label, score in zip(prediction["boxes"][keep], prediction["labels"][keep],
                                         prediction["scores"][keep]):
                x1, y1, x2, y2 = box.tolist()
                axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         fill=False, edgecolor="red", linewidth=2))
                axis.text(x1, y1, f"{categories[int(label)]} {float(score):.2f}",
                          fontsize=8, color="white", bbox={"facecolor": "red", "alpha": 0.7})
            count = int(keep.sum())
            axis.set_title(f"Threshold={threshold:g}, boxes={count}")
            axis.axis("off")
            rows.append([path.name, threshold, count, round(elapsed_ms, 2)])
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{index:02d}_{path.stem}_comparison.png", dpi=120)
        plt.close(fig)
        print(f"{path.name}：推理 {elapsed_ms:.1f} ms", flush=True)
    with (OUTPUT_DIR / "results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "threshold", "box_count", "shared_inference_ms"])
        writer.writerows(rows)
    print(f"结果已保存到 {OUTPUT_DIR}；请观察误检、漏检与阈值的关系。")


if __name__ == "__main__":
    main()
