"""功能检查用例；临时文本仅测试代码逻辑，不产生实验性能指标。"""

import copy
from contextlib import redirect_stdout
import csv
import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

from models import SentimentClassifier, model_from_config
from prepare_data import (SentimentDataset, balanced_sample, build_vocab,
                          deduplicate, encode_text, load_prepared, read_reviews)
from train import train_epoch, smoke_batch
from utils import (check_same_conditions, FAIR_KEYS, set_seed, write_json,
                   sha256_file, object_hash, evaluate_loader)


class DataTests(unittest.TestCase):
    def test_encoding_unk_padding_and_truncation(self):
        vocab = build_vocab([{"text": "不好！"}])
        self.assertEqual(vocab["<PAD>"], 0)
        self.assertEqual(vocab["<UNK>"], 1)
        ids, length = encode_text("好？", vocab, max_length=64)
        self.assertEqual(ids[:2], [vocab["好"], 1])
        self.assertEqual(length, 2)
        self.assertEqual(ids[2:], [0] * 62)
        self.assertEqual(encode_text("好" * 80, vocab, max_length=64)[1], 64)
        self.assertNotIn("？", vocab)  # 验证/测试的新字符不能进入训练字符表。

    def test_longer_input_keeps_characters_beyond_old_limit(self):
        rows = [{"text": "好" * 80 + "差" * 240, "label": 0}]
        vocab = build_vocab(rows)
        dataset = SentimentDataset(rows, vocab, max_length=256)
        self.assertEqual(tuple(dataset.inputs.shape), (1, 256))
        self.assertEqual(dataset.lengths.tolist(), [256])
        self.assertTrue((dataset.inputs[0, 80:] == vocab["差"]).all())
        ids, length = encode_text("好" * 80, vocab, max_length=256)
        self.assertEqual(length, 80)
        self.assertEqual(ids[80:], [0] * 176)

    def test_reader_rejects_bad_rows_and_excludes_neutral(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows([["id", "review", "star"], [1, "不错，不贵！", "5.0"],
                                  [2, "不好", "1.0"], [3, "一般", 3], [4, "  ", 4],
                                  [5, "无效", "nan"], [6, "无效", "abc"], [7, "无效", "2.5"]])
            rows, stats = read_reviews(path)
            self.assertEqual([row["label"] for row in rows], [1, 0])
            self.assertEqual(rows[0]["text"], "不错，不贵！")
            self.assertEqual(stats["empty_text"], 1)
            self.assertEqual(stats["invalid_rating"], 3)
            self.assertEqual(stats["excluded_neutral"], 1)

    def test_global_duplicates_and_conflicting_labels(self):
        splits = {"train": [{"text": "相同", "label": 0}, {"text": "冲突", "label": 0}],
                  "val": [{"text": "相同", "label": 0}, {"text": "独立", "label": 1}],
                  "test": [{"text": "冲突", "label": 1}, {"text": "测试", "label": 0}]}
        cleaned, _ = deduplicate(splits)
        self.assertEqual([r["text"] for r in cleaned["train"]], ["相同"])
        self.assertEqual([r["text"] for r in cleaned["val"]], ["独立"])
        self.assertEqual([r["text"] for r in cleaned["test"]], ["测试"])

    def test_sampling_is_repeatable_and_balanced(self):
        rows = [{"text": str(i), "label": i % 2} for i in range(40)]
        sampled = balanced_sample(rows, 20, 42)
        self.assertEqual(sampled, balanced_sample(rows, 20, 42))
        self.assertEqual(sum(r["label"] for r in sampled), 10)
        self.assertEqual(len({r["text"] for r in sampled}), 20)

    def test_batch_shapes_and_lengths(self):
        rows = [{"text": "好", "label": 1}, {"text": "不好" * 40, "label": 0}]
        dataset = SentimentDataset(rows, build_vocab(rows), max_length=64)
        inputs, labels, lengths = next(iter(DataLoader(dataset, batch_size=2)))
        self.assertEqual(tuple(inputs.shape), (2, 64))
        self.assertEqual(tuple(labels.shape), (2,))
        self.assertEqual(lengths.tolist(), [1, 64])
        self.assertTrue((inputs[0, 1:] == 0).all())

    def test_detects_modified_processed_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "vocab.json", {"<PAD>": 0, "<UNK>": 1})
            manifest = {"processed_sha256": {"vocab.json": sha256_file(root / "vocab.json")}}
            manifest["data_id"] = object_hash(manifest)
            write_json(root / "manifest.json", manifest)
            load_prepared(root)
            write_json(root / "vocab.json", {"changed": 1})
            with self.assertRaises(ValueError):
                load_prepared(root)

    def test_detects_modified_data_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"max_length": 64, "processed_sha256": {}}
            manifest["data_id"] = object_hash(manifest)
            manifest["max_length"] = 32
            write_json(root / "manifest.json", manifest)
            with self.assertRaises(ValueError):
                load_prepared(root)


class ModelTests(unittest.TestCase):
    def setUp(self):
        set_seed(42, 2)

    def test_last_valid_step_is_independent_of_padding(self):
        # 同一有效前缀，右侧换成非零 ID 仍应输出相同 logits。
        for kind, pooling in ((k, p) for k in ("rnn", "lstm") for p in ("last", "mean")):
            with self.subTest(model=kind, pooling=pooling):
                model = SentimentClassifier(12, kind, pooling=pooling, dropout=0.2)
                model.eval()
                inputs = torch.tensor([[2, 3, 4, 0, 0], [5, 0, 0, 0, 0]])
                lengths = torch.tensor([3, 1])
                altered = torch.tensor([[2, 3, 4, 8, 9], [5, 6, 7, 8, 9]])
                with torch.no_grad():
                    expected = torch.cat([model(inputs[i:i+1, :length], torch.tensor([length]))
                                          for i, length in enumerate(lengths.tolist())])
                    actual = model(inputs, lengths)
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(actual, model(altered, lengths), atol=1e-6, rtol=1e-5)

    def test_backward_update_and_checkpoint_reload(self):
        rows = [{"text": "好！", "label": 1}, {"text": "不好", "label": 0}]
        vocab = build_vocab(rows)
        loader = DataLoader(SentimentDataset(rows, vocab), batch_size=2)
        inputs, _, lengths = next(iter(loader))
        for kind, pooling in ((k, p) for k in ("rnn", "lstm") for p in ("last", "mean")):
            with self.subTest(model=kind, pooling=pooling):
                config = {"vocab_size": len(vocab), "model_type": kind,
                          "embedding_dim": 32, "hidden_dim": 64}
                # 旧配置没有新增字段；验证默认回退仍可重载旧模型。
                if pooling == "mean":
                    config.update(pooling="mean", dropout=0.2)
                model = model_from_config(config)
                before = copy.deepcopy(model.state_dict())
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                loss = train_epoch(model, loader, optimizer, grad_clip=0.1)
                self.assertTrue(0 < loss < float("inf"))
                self.assertFalse(torch.equal(before["classifier.weight"], model.classifier.weight))
                self.assertEqual(tuple(model(inputs, lengths).shape), (2, 2))
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                norm = torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None))
                self.assertLessEqual(norm.item(), 0.10001)
                self.assertTrue((model.embedding.weight[0] == 0).all())
                model.eval()
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "model.pt"
                    torch.save(model.state_dict(), path)
                    reloaded = model_from_config(config)
                    reloaded.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
                    reloaded.eval()
                    torch.testing.assert_close(model(inputs, lengths), reloaded(inputs, lengths))
                result = evaluate_loader(model, loader)
                self.assertEqual(result["total"], 2)

    def test_smoke_check_preserves_weights_and_dropout_random_state(self):
        rows = [{"text": "好！", "label": 1}, {"text": "不好", "label": 0}]
        vocab = build_vocab(rows)
        batch = next(iter(DataLoader(SentimentDataset(rows, vocab), batch_size=2)))
        model = SentimentClassifier(len(vocab), pooling="mean", dropout=0.4)
        before_weights = copy.deepcopy(model.state_dict())
        before_random = torch.get_rng_state().clone()
        with redirect_stdout(io.StringIO()):
            smoke_batch(model, batch, 0.002)
        self.assertTrue(torch.equal(before_random, torch.get_rng_state()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before_weights[key], rtol=0, atol=0)

    def test_comparison_rejects_different_conditions(self):
        first = {key: 1 for key in FAIR_KEYS}
        for key in ("epochs", "pooling", "dropout", "weight_decay", "grad_clip"):
            with self.subTest(condition=key):
                second = dict(first, **{key: 2})
                with self.assertRaises(ValueError):
                    check_same_conditions([first, second])


if __name__ == "__main__":
    unittest.main(verbosity=2)
