"""下载真实 ASAP 数据，保留官方划分，建立训练集字符表及定长编码。"""

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import random
from urllib.error import URLError
from urllib.request import urlopen

import torch
from torch.utils.data import Dataset, DataLoader

from utils import ROOT, DATA_DIR, read_json, write_json, sha256_file, object_hash

REVISION = "975122a60065240124df62cb4d5dbfd19ed9ef2c"
SOURCE_URL = f"https://raw.githubusercontent.com/Meituan-Dianping/asap/{REVISION}/"
RAW_FILES = {
    "train.csv": ("data/train.csv", "30912f5e2a866a4c0f654ec3461215f008b8b90d208f2c410f03369f43655de9"),
    "dev.csv": ("data/dev.csv", "54efc8f6bc26d63a9f87e3418cd68e47f2eb13a19085e85abe63f3eafafe2b44"),
    "test.csv": ("data/test.csv", "217dd27c1d5699067952474e40824935d56ecf8b5047950d6da3d1222976a2bd"),
    "LICENSE-ASAP.txt": ("LICENSE", "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
    "README-ASAP.md": ("README.md", "076517c161644c6fc5b08dbd15c3508950c05ed857f683e073fbd9ed51821208"),
}
MAX_LENGTH = 256
PAD_ID, UNK_ID = 0, 1
# 点击运行时使用这些默认参数；需要调整时修改并保存后重新运行。
SAMPLES = 2500
SEED = 42
OFFLINE = False


def download_raw(raw_dir, offline=False):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, (remote, expected) in RAW_FILES.items():
        path = raw_dir / name
        if not path.exists():
            if offline:
                raise FileNotFoundError(f"缺少 {path}；下载地址：{SOURCE_URL + remote}")
            print(f"下载 {name} ...", flush=True)
            temporary = path.with_suffix(path.suffix + ".part")
            try:
                # 每个文件只尝试一次；超时或失败时给出手动下载说明。
                with urlopen(SOURCE_URL + remote, timeout=45) as response:
                    with temporary.open("wb") as handle:
                        while block := response.read(1024 * 1024):
                            handle.write(block)
                if sha256_file(temporary) != expected:
                    raise ValueError(f"{name} 校验失败，请按 data/README.md 重新下载")
                temporary.replace(path)
            except (URLError, OSError, ValueError) as exc:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"下载失败：{exc}\n地址：{SOURCE_URL + remote}\n"
                                   f"请手动保存到 {path}，详见 data/README.md，"
                                   "随后将本文件顶部的 OFFLINE 改为 True，再点击运行") from exc
        if sha256_file(path) != expected:
            raise ValueError(f"{path} 与固定版本的 SHA256 不符，请重新下载；不会覆盖已有文件")


def read_reviews(path):
    rows, stats = [], Counter()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"id", "review", "star"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} 应包含 id,review,star 表头，请下载官方完整 CSV")
        for row in reader:
            stats["raw_rows"] += 1
            text = (row.get("review") or "").strip()
            if not text:
                stats["empty_text"] += 1
                continue
            try:
                star = float(row["star"])
            except (TypeError, ValueError):
                star = -1
            if star not in (1, 2, 3, 4, 5):
                stats["invalid_rating"] += 1
                continue
            if star == 3:
                stats["excluded_neutral"] += 1
                continue
            rows.append({"id": row["id"], "text": text, "rating": int(star),
                         "label": int(star >= 4)})
    stats["valid_binary_rows"] = len(rows)
    stats["negative_before_dedup"] = sum(row["label"] == 0 for row in rows)
    stats["positive_before_dedup"] = sum(row["label"] == 1 for row in rows)
    return rows, dict(stats)


def deduplicate(splits):
    """先丢弃标签冲突文本，再全局去重；保留第一次出现的官方归属。"""
    labels_by_text = defaultdict(set)
    for rows in splits.values():
        for row in rows:
            labels_by_text[row["text"]].add(row["label"])
    conflicts = {text for text, labels in labels_by_text.items() if len(labels) > 1}
    seen, cleaned, stats = set(), {}, {}
    for split in ("train", "val", "test"):
        kept, removed = [], Counter()
        for row in splits[split]:
            text = row["text"]
            if text in conflicts:
                removed["conflicting_rows_removed"] += 1
            elif text in seen:
                removed["duplicate_rows_removed"] += 1
            else:
                seen.add(text)
                kept.append(row)
        cleaned[split], stats[split] = kept, dict(removed)
    return cleaned, stats


def balanced_sample(rows, size, seed):
    """在每个官方子集中按标签分层、无放回抽取均衡教学样本。"""
    rng, sampled = random.Random(seed), []
    for label, quota in ((0, size // 2), (1, size - size // 2)):
        candidates = [row for row in rows if row["label"] == label]
        if len(candidates) < quota:
            raise ValueError(f"标签 {label} 仅有 {len(candidates)} 条，无法抽取 {quota} 条；"
                             "请减小 --samples")
        sampled.extend(rng.sample(candidates, quota))
    rng.shuffle(sampled)
    return sampled


def build_vocab(train_rows):
    # 只看训练集；排序保证相同数据在重复运行时得到相同字符 ID。
    characters = sorted({char for row in train_rows for char in row["text"]})
    return {"<PAD>": PAD_ID, "<UNK>": UNK_ID,
            **{char: index + 2 for index, char in enumerate(characters)}}


def encode_text(text, vocab, max_length=MAX_LENGTH):
    if not text or max_length < 1:
        raise ValueError("编码需要非空文本和正的最大长度")
    ids = [vocab.get(char, UNK_ID) for char in text[:max_length]]
    length = len(ids)  # 截断后的有效长度，不包含右侧 PAD。
    return ids + [PAD_ID] * (max_length - length), length


class SentimentDataset(Dataset):
    def __init__(self, rows, vocab, max_length=MAX_LENGTH):
        self.rows = rows
        encoded = [encode_text(row["text"], vocab, max_length) for row in rows]
        self.inputs = torch.tensor([item[0] for item in encoded], dtype=torch.long)
        self.lengths = torch.tensor([item[1] for item in encoded], dtype=torch.long)
        self.labels = torch.tensor([row["label"] for row in rows], dtype=torch.long)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        # input: [L]，label 与 length: 标量；DataLoader 会添加批次维 N。
        return self.inputs[index], self.labels[index], self.lengths[index]


def load_prepared(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    if not (data_dir / "manifest.json").exists():
        raise FileNotFoundError("尚未准备数据，请先打开 prepare_data.py，点击右上角运行")
    manifest = read_json(data_dir / "manifest.json")
    identity = {key: value for key, value in manifest.items() if key != "data_id"}
    if manifest.get("data_id") != object_hash(identity):
        raise ValueError("数据配置已改变，请重新准备数据并重新训练两模型")
    # 即使有人手动改了数据，也不能悄悄沿用旧模型结果。
    for name, expected in manifest["processed_sha256"].items():
        if sha256_file(data_dir / name) != expected:
            raise ValueError(f"{name} 已改变，请重新准备数据并重新训练两模型")
    return read_json(data_dir / "vocab.json"), manifest


def prepare(samples=2500, seed=42, raw_dir=ROOT / "data" / "raw",
            data_dir=DATA_DIR, offline=False, max_length=MAX_LENGTH):
    if samples < 20:
        raise ValueError("--samples 至少为 20，建议 2000–3000")
    if max_length < 1:
        raise ValueError("最大文本长度必须为正整数")
    download_raw(raw_dir, offline)
    splits, statistics = {}, {}
    for split, filename in (("train", "train.csv"), ("val", "dev.csv"), ("test", "test.csv")):
        splits[split], statistics[split] = read_reviews(Path(raw_dir) / filename)
    splits, removed = deduplicate(splits)
    sizes = {"train": samples * 8 // 10, "val": samples // 10}
    sizes["test"] = samples - sizes["train"] - sizes["val"]
    for offset, split in enumerate(("train", "val", "test")):
        statistics[split].update(removed[split])
        statistics[split]["eligible_after_dedup"] = len(splits[split])
        splits[split] = balanced_sample(splits[split], sizes[split], seed + offset)
        for row in splits[split]:
            row["length"] = min(len(row["text"]), max_length)
        counts = Counter(row["label"] for row in splits[split])
        statistics[split].update({"selected": len(splits[split]), "negative": counts[0],
                                  "positive": counts[1],
                                  "truncated": sum(len(r["text"]) > max_length for r in splits[split])})
        print(f"{split}: {statistics[split]}")
    texts = [set(row["text"] for row in splits[s]) for s in ("train", "val", "test")]
    if texts[0] & texts[1] or texts[0] & texts[2] or texts[1] & texts[2]:
        raise ValueError("检测到跨集合重复文本")
    vocab = build_vocab(splits["train"])
    data_dir = Path(data_dir)
    for split, rows in splits.items():
        write_json(data_dir / f"{split}.json", rows)
    write_json(data_dir / "vocab.json", vocab)
    files = ("train.json", "val.json", "test.json", "vocab.json")
    manifest = {"dataset": "ASAP-rating-binary", "revision": REVISION,
                "source": "https://github.com/Meituan-Dianping/asap", "license": "Apache-2.0",
                "label_rule": "1–2 stars -> 0; 4–5 stars -> 1; exclude 3 stars",
                "sampling": "balanced within official splits; no reassignment",
                "seed": seed, "samples": samples, "max_length": max_length,
                "statistics": statistics,
                "raw_sha256": {name: sha256_file(Path(raw_dir) / name) for name in RAW_FILES},
                "processed_sha256": {name: sha256_file(data_dir / name) for name in files}}
    manifest["data_id"] = object_hash(manifest)
    write_json(data_dir / "manifest.json", manifest)
    dataset = SentimentDataset(splits["train"], vocab, max_length)
    inputs, labels, lengths = next(iter(DataLoader(dataset, batch_size=32)))
    example = {"text": splits["train"][0]["text"], "visible_text": splits["train"][0]["text"][:max_length],
               "characters": list(splits["train"][0]["text"][:max_length]),
               "input_ids": inputs[0].tolist(), "effective_length": lengths[0].item(),
               "label": labels[0].item()}
    write_json(data_dir / "encoding_example.json", example)
    print(f"字符表大小：{len(vocab)}，PAD=0，UNK=1，仅来自训练集")
    print("真实编码示例：", example)
    print(f"批次输入 {list(inputs.shape)}，标签 {list(labels.shape)}，有效长度 {list(lengths.shape)}")
    print("三个集合的重复文本交集均为 0；数据 ID：", manifest["data_id"])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--offline", action="store_true", default=OFFLINE,
                        help="只读取已手动放置的官方原始文件")
    args = parser.parse_args()
    try:
        prepare(args.samples, args.seed, args.raw_dir, args.data_dir, args.offline, args.max_length)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"数据准备失败：{exc}\n")


if __name__ == "__main__":
    main()
