"""点击右上角运行即可训练 LSTM；RNN 与单批次检查入口共用本文件。"""

import argparse
import copy
import platform
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models import model_from_config
from prepare_data import SentimentDataset, load_prepared
from utils import (DATA_DIR, OUTPUT_DIR, set_seed, read_json, write_json, write_csv,
                   count_parameters, evaluate_loader, sha256_file)


# 演示参数：修改后保存文件，再点击右上角的“运行 Python 文件”。
# RNN 训练与两种模型的检查入口共用以下超参数，保证比较条件一致。
MODEL = "lstm"       # 本文件直接运行时训练 LSTM；train_rnn.py 固定使用 RNN。
EPOCHS = 20
BATCH_SIZE = 32
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
LEARNING_RATE = 0.002
POOLING = "last"     # 用最后有效状态概括前缀，比较信息沿序列传递的效果。
# "mean" 可直接汇总各位置线索，作为另一种表示方式对照。
DROPOUT = 0.2
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0      # 限制梯度范数；0 表示关闭。
SEED = 42
THREADS = 2


def train_epoch(model, loader, optimizer, grad_clip=GRAD_CLIP):
    model.train()
    criterion, loss_sum, total = nn.CrossEntropyLoss(), 0.0, 0
    for inputs, labels, lengths in loader:
        optimizer.zero_grad()
        logits = model(inputs, lengths)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("训练 Loss 不是有限值，请检查输入和学习率")
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        total += labels.size(0)
    return loss_sum / total


def smoke_batch(model, batch, learning_rate, grad_clip=GRAD_CLIP):
    # 副本隔离参数，fork_rng 隔离 Dropout 随机数，正式训练不受检查影响。
    with torch.random.fork_rng(devices=[]):
        trial = copy.deepcopy(model)
        trial.train()
        inputs, labels, lengths = batch
        optimizer = torch.optim.Adam(trial.parameters(), lr=learning_rate)
        before = trial.classifier.weight.detach().clone()
        embedded = trial.embedding(inputs)
        outputs, _ = trial.recurrent(embedded)
        logits = trial(inputs, lengths)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in trial.parameters()):
            raise RuntimeError("单批次检查发现非有限梯度")
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(trial.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if not torch.isfinite(loss) or torch.equal(before, trial.classifier.weight):
            raise RuntimeError("单批次检查失败：Loss 无效或参数没有更新")
        print(f"张量形状：输入 {list(inputs.shape)} -> Embedding {list(embedded.shape)}"
              f" -> 循环输出 {list(outputs.shape)} -> logits {list(logits.shape)}")
        print(f"单批次前向/反向/Adam 更新成功，Loss={loss.item():.4f}（使用独立模型副本）")


def plot_history(history, path, model_type):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(epochs, [row["train_loss"] for row in history], "o-")
    axes[0].set(xlabel="Epoch", ylabel="Training loss", title=model_type.upper())
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], "o-")
    axes[1].set(xlabel="Epoch", ylabel="Validation accuracy", ylim=(0, 1))
    for ax in axes:
        ax.set_xticks(epochs)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(args):
    if min(args.epochs, args.batch_size, args.embedding_dim, args.hidden_dim) < 1:
        raise ValueError("Epoch、批次大小及模型维度必须为正整数")
    if not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("学习率必须为正的有限值")
    if not np.isfinite(args.dropout) or not 0 <= args.dropout < 1:
        raise ValueError("Dropout 必须在 [0, 1) 范围内")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("权重衰减必须为非负的有限值")
    if not np.isfinite(args.grad_clip) or args.grad_clip < 0:
        raise ValueError("梯度裁剪阈值必须为非负的有限值")
    set_seed(args.seed, args.threads)
    vocab, manifest = load_prepared(args.data_dir)
    config = {"model_type": args.model, "vocab_size": len(vocab),
              "embedding_dim": args.embedding_dim, "hidden_dim": args.hidden_dim,
              "num_layers": 1, "bidirectional": False,
              "max_length": manifest["max_length"], "epochs": args.epochs,
              "batch_size": args.batch_size, "learning_rate": args.lr,
              "seed": args.seed, "threads": args.threads, "device": "cpu",
              "pooling": args.pooling, "dropout": args.dropout,
              "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
              "optimizer": "Adam", "loss": "CrossEntropyLoss", "data_id": manifest["data_id"]}
    train_data = SentimentDataset(read_json(args.data_dir / "train.json"), vocab, config["max_length"])
    val_data = SentimentDataset(read_json(args.data_dir / "val.json"), vocab, config["max_length"])
    # 单独的生成器保证两模型各 Epoch 的训练样本顺序一致。
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=generator, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = model_from_config(config).cpu()
    print(model)
    parameter_count = count_parameters(model)
    print(f"CPU；线程数={args.threads}；可训练参数量={parameter_count:,}")
    print(f"最大长度={config['max_length']}；汇总方式={args.pooling}；Dropout={args.dropout}")
    check_loader = DataLoader(train_data, batch_size=args.batch_size,
                              generator=torch.Generator().manual_seed(args.seed))
    smoke_batch(model, next(iter(check_loader)), args.lr, args.grad_clip)
    if args.check_only:
        print("单批次检查完成，不保存训练结果，不读取测试样本。")
        return
    destination = args.output_dir / args.model
    destination.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history, best_accuracy, best_epoch, training_seconds = [], -1.0, 0, 0.0
    checkpoint_path = destination / "best.pt"
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, args.grad_clip)
        elapsed = time.perf_counter() - start
        training_seconds += elapsed
        # 计时只累计训练批次（含加载、前向、反向、更新）；不含验证、保存和绘图。
        validation = evaluate_loader(model, val_loader)
        row = {"epoch": epoch, "train_loss": train_loss,
               "val_loss": validation["loss"], "val_accuracy": validation["accuracy"],
               "train_seconds": elapsed}
        history.append(row)
        print(f"Epoch {epoch}/{args.epochs} | 训练 Loss {train_loss:.4f} | "
              f"验证 Accuracy {validation['accuracy']:.4f} | 训练 {elapsed:.3f}s", flush=True)
        # 相同验证准确率保留较早轮次，不查看测试结果。
        if validation["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = validation["accuracy"], epoch
            torch.save({"state_dict": model.state_dict(), "config": config, "vocab": vocab,
                        "best_epoch": epoch, "best_val_accuracy": best_accuracy,
                        "data_id": manifest["data_id"]}, checkpoint_path)
    write_csv(destination / "history.csv", history, list(history[0]))
    write_json(destination / "config.json", config)
    summary = {"model": args.model, "config": config, "parameters": parameter_count,
               "best_epoch": best_epoch, "best_val_accuracy": best_accuracy,
               "training_seconds": training_seconds,
               "timing_scope": "sum of training epochs only; excludes validation, smoke check, saving, plotting, download",
               "checkpoint_sha256": sha256_file(checkpoint_path),
               "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                               "numpy": np.__version__, "platform": platform.platform()}}
    write_json(destination / "summary.json", summary)
    plot_history(history, destination / "training_curves.png", args.model)
    print(f"最佳验证轮次 {best_epoch}，Accuracy={best_accuracy:.4f}；累计训练 {training_seconds:.3f}s")
    print(f"模型、日志、配置和曲线已保存：{destination}")


def main(argv=None, *, default_model=MODEL, check_only=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("lstm", "rnn"), default=default_model)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--embedding-dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--pooling", choices=("last", "mean"), default=POOLING)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--check-only", action="store_true", default=check_only,
                        help="只检查结构和一个批次的训练")
    args = parser.parse_args(argv)
    try:
        train(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"训练失败：{exc}\n")


if __name__ == "__main__":
    main()
