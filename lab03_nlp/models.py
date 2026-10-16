"""两个模型只替换循环层，复用字符嵌入和分类代码。"""

import torch
from torch import nn


class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, model_type="lstm", embedding_dim=32, hidden_dim=64,
                 pooling="last", dropout=0.0):
        super().__init__()
        if model_type not in ("rnn", "lstm"):
            raise ValueError("model_type 必须为 rnn 或 lstm")
        if pooling not in ("last", "mean"):
            raise ValueError("pooling 必须为 last 或 mean")
        if not 0 <= dropout < 1:
            raise ValueError("dropout 必须在 [0, 1) 范围内")
        self.pooling = pooling
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        recurrent = nn.LSTM if model_type == "lstm" else nn.RNN
        # 默认单层、单向；batch_first=True 表示批次维 N 位于第 0 维。
        self.recurrent = recurrent(embedding_dim, hidden_dim, num_layers=1,
                                   batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, inputs, lengths):
        # inputs: [N,L]；每个字符 ID 查表得到 E 维可训练向量。
        embedded = self.dropout(self.embedding(inputs))
        outputs, _ = self.recurrent(embedded)
        # [N,L,H]：每个时间步的隐藏表示。单向网络的有效部分不受右侧 PAD 影响。
        if self.pooling == "mean":
            # 汇总全部有效位置的情感线索；PAD 对分子与分母都没有贡献。
            positions = torch.arange(inputs.size(1), device=outputs.device)
            mask = positions[None, :] < lengths[:, None]
            representation = (outputs * mask.unsqueeze(-1)).sum(dim=1) / lengths[:, None]
        else:
            # 最后有效步表示；早期信息需要经循环状态传递，不能取右侧 PAD。
            batch_indices = torch.arange(inputs.size(0), device=outputs.device)
            representation = outputs[batch_indices, lengths - 1]
        # 线性层把 H 维表示映射为两类分数 [N,2]。
        # CrossEntropyLoss 接收 logits，因此这里不加 softmax。
        return self.classifier(self.dropout(representation))


def model_from_config(config):
    return SentimentClassifier(config["vocab_size"], config["model_type"],
                               config["embedding_dim"], config["hidden_dim"],
                               pooling=config.get("pooling", "last"),
                               dropout=config.get("dropout", 0.0))
