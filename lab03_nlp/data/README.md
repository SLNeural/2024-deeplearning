# 数据来源、许可和手动准备

## 来源与教学转换

数据集：**ASAP — A Chinese Review Dataset Towards Aspect Category Sentiment Analysis and Rating Prediction**。

- 作者数据仓库：https://github.com/Meituan-Dianping/asap
- 论文原始发布：https://aclanthology.org/2021.naacl-main.167/
- 原始文本：大众点评真实中文餐饮评论；原始规模 46,730 条。
- 官方划分：train 36,850 / dev 4,940 / test 4,940。
- 发布仓库许可：[Apache License 2.0](https://github.com/Meituan-Dianping/asap/blob/975122a60065240124df62cb4d5dbfd19ed9ef2c/LICENSE)。本目录的 `raw/LICENSE-ASAP.txt` 保留完整原许可，`raw/README-ASAP.md` 保留作者说明与引用。复用或分发数据时应保留来源及许可证。
- 固定源版本：`975122a60065240124df62cb4d5dbfd19ed9ef2c`。

本实验只用 `review` 和整体 `star`，使用 `id` 追溯样本：1–2 星→0（负面），4–5 星→1（正面），3 星排除。不使用 18 个方面的细粒度标签。这是由真实整体评分派生的二分类任务，不能视为每个句子都有独立人工情感标注。

ASAP 评论较长。优化版将最大输入长度从 64 提高到 256，保留同一批 2500 条样本及标签。当前有 1473 条超过 256 字（训练/验证/测试为 1164/160/149）。保存完整原文，编码时只读前 256 字，报告仍需说明截断损失与整体星级/局部情绪不一致的影响。长度可在 `prepare_data.py` 顶部的 `MAX_LENGTH` 调整，随后重新准备数据并训练两模型。

## 自动方式

在 VS Code 中打开 `lab03_nlp/prepare_data.py`，点击右上角的“运行 Python 文件”三角按钮。抽样数量和随机种子可在文件顶部设置。

首次下载约 46 MB；每文件仅尝试一次，超时 45 秒。已有文件使用本地缓存并检查 SHA256，不覆盖校验失败的文件。

## 手动方式

若当前机器不能访问数据源，可以在可访问 GitHub 的浏览器/机器下载以下 5 个文件并复制到 `lab03_nlp/data/raw/`。不要保存 GitHub 页面 HTML，不要用 `*_sample.csv` 替代完整文件。

| 本地文件名 | 固定下载地址 |
|---|---|
| `train.csv` | https://raw.githubusercontent.com/Meituan-Dianping/asap/975122a60065240124df62cb4d5dbfd19ed9ef2c/data/train.csv |
| `dev.csv` | https://raw.githubusercontent.com/Meituan-Dianping/asap/975122a60065240124df62cb4d5dbfd19ed9ef2c/data/dev.csv |
| `test.csv` | https://raw.githubusercontent.com/Meituan-Dianping/asap/975122a60065240124df62cb4d5dbfd19ed9ef2c/data/test.csv |
| `LICENSE-ASAP.txt` | https://raw.githubusercontent.com/Meituan-Dianping/asap/975122a60065240124df62cb4d5dbfd19ed9ef2c/LICENSE |
| `README-ASAP.md` | https://raw.githubusercontent.com/Meituan-Dianping/asap/975122a60065240124df62cb4d5dbfd19ed9ef2c/README.md |

文件为 UTF-8 CSV，表头包含 `id,review,star`，后面还有方面标签。评分在原始 CSV 中写成 `1.0`–`5.0`；文本中的逗号、换行由 CSV 引号规则处理，程序使用 Python 标准库 `csv.DictReader` 读取，无需 pandas 或手动改格式。保留原始文件，程序会核对 `prepare_data.py` 中固定的 SHA256。

文件复制完后，将 `prepare_data.py` 顶部的 `OFFLINE = False` 改为 `OFFLINE = True`，保存并点击右上角运行。

## 处理规则与可复现信息

1. 检查空文本与无效星级，统计原始正负类别；排除 3 星。
2. 文本只去除首尾空白；保留“不”“没”、标点等。
3. 同一文本若二分类标签冲突，移除全部冲突记录；其余重复文本只留一条，按 train→val→test 优先级保持第一次出现的官方归属。
4. 不混合官方划分。默认各集合内按固定种子、类别配额无放回抽取 2000/250/250 条，各正负一半。程序检查三个集合的全文交集为零。均衡抽样改变了原始类别先验，不代表平台实际类别比例。
5. 仅训练全文建立字符表，排序确定 ID；PAD=0，UNK=1。编码时截取前 256 字并右填充；每条样本的有效长度保存在处理后 JSON 的 `length` 字段，Dataset 编码时同样计算并保存为长度张量。数据标识包含实际最大长度，改变长度会要求重新训练。
6. 保存清洗统计、原始和处理后文件校验值、来源版本、最大长度及唯一数据 ID 到 `processed/manifest.json`。`encoding_example.json` 给出一条真实文本的实际编码。

只排除完全相同（去首尾空白后）的全文重复，不声称解决近重复文本或截断后相同前缀。训练脚本只用训练/验证指标；为确认数据完整性会校验所有处理文件的哈希，不计算测试性能。

没有官方划分的数据需要另做固定种子 42 的分层划分；本项目选定的数据自带官方划分，因此无需这一分支，不提供额外的复杂通用数据适配器。

## 学术引用

Bu, Jiahao; Ren, Lei; Zheng, Shuang; Yang, Yang; Wang, Jingang; Zhang, Fuzheng; Wu, Wei. 2021. *ASAP: A Chinese Review Dataset Towards Aspect Category Sentiment Analysis and Rating Prediction*. NAACL-HLT, 2069–2079. https://aclanthology.org/2021.naacl-main.167/
