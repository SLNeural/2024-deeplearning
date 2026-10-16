# 实验项目三：中文文本情感分类与 RNN/LSTM 比较

面向人工智能专业大三《深度学习与应用》，共两次实验、8 学时。使用字符编码、Embedding、单向单层 RNN/LSTM、最后有效状态分类，不需要分词器、预训练模型、联网推理服务或显卡。

## 1. 文件与流程

```text
lab03_nlp/
├── requirements.txt         # 仅 torch、numpy、matplotlib
├── prepare_data.py          # 下载、检查、去重、抽样、字符表、Dataset
├── models.py                # Embedding → RNN/LSTM → 最后有效状态 → Linear
├── check_lstm.py            # 点击运行：LSTM 结构和单批次检查
├── train.py                 # 点击运行：训练 LSTM；顶部集中设置共用参数
├── check_rnn.py             # 点击运行：RNN 结构和单批次检查
├── train_rnn.py             # 点击运行：使用相同参数训练 RNN
├── evaluate.py              # 同一测试集最终评价、真实误分类导出
├── compare.py               # 指标表、比较图、训练曲线
├── utils.py                 # 随机种子、文件读写、共用评价函数
├── test_pipeline.py         # 教师/开发者功能检查
├── data/README.md           # 来源、许可、手动下载、格式说明
├── data/raw/                # 官方 CSV、原始许可与说明
├── data/processed/          # 划分、字符表、统计与编码示例
├── outputs/                 # 实际运行结果
└── docs/
    ├── 实验指导书.md
    ├── 实验报告模板.md
    └── 验证记录.md
```

## 2. 数据与边界

采用作者发布的 [ASAP](https://github.com/Meituan-Dianping/asap)，原始来源为大众点评真实中文餐饮评论，发布仓库包含 [Apache-2.0 许可证](https://github.com/Meituan-Dianping/asap/blob/975122a60065240124df62cb4d5dbfd19ed9ef2c/LICENSE)。论文：[Bu 等，NAACL 2021](https://aclanthology.org/2021.naacl-main.167/)。选择该数据是因为作者来源、许可和 CSV 下载均可直接核实。

本实验从整体星级派生二分类：1–2 星为负面（0），4–5 星为正面（1），排除 3 星；不使用细粒度方面标签。星级标签与局部句子情绪可能不一致。保留官方 train/dev/test 归属，先检查无效记录、移除标签冲突文本、全局去重，再以种子 42 在各集合内按类别均衡抽样。默认共 2500 条（2000/250/250），不重新混合官方集合。此均衡子集的类别分布与原始平台不同，Accuracy 只代表当前教学子集。

仅去除文本首尾空白，保留否定词、标点、内部空格与 emoji。字符表仅由训练集建立，PAD=0、UNK=1，长度固定 256，右侧填充，并保存截断后的有效长度。默认 2500 条中有 **1473 条超过 256 字**（训练/验证/测试为 1164/160/149）；模型研究的是评论前 256 字的分类。误分类文件同时提供完整文本和实际可见前缀，不能把前缀分类描述成完整评论的理解能力。

当前实验保留原有 2000/250/250 条样本的原文、标签、顺序、输入长度和字符表，使用最后有效状态表示前 256 字，训练预算为 20 Epoch。早期位置的信息需要通过循环状态传递到末端，因此这个表示方式适合观察 RNN/LSTM 的序列信息传递差异。两模型共用全部训练设置，并各自按最佳验证准确率选模型；测试集不参与候选选择。

上一版平均池化的代码、数据和完整结果保存在 `outputs/baseline_256_mean/`。两种表示方式的准确率都应如实报告，最后有效状态不保证比平均池化更准确，也不保证 LSTM 在所有数据和随机种子上更好。本次方案与验证记录见 `outputs/last_state_experiment/`。更早的 64 字版归档在 `outputs/baseline_64chars/`，其优化记录在 `outputs/optimization/`。

当前服务器实际运行：LSTM 测试准确率 **81.2%（203/250）**，RNN **73.2%（183/250）**；最佳验证准确率分别为 80.8% 和 74.8%。旧平均池化版的测试准确率分别为 86.8% 和 89.2%。当前版更适合展示结构差异，两个模型的绝对准确率都降低了；两版同时改变了池化方式和训练预算，不能单独归因于某一项。版本对比详见 `outputs/last_state_experiment/before_after.md`。

下载固定在版本 `975122a60065240124df62cb4d5dbfd19ed9ef2c`，校验每个原始文件的 SHA256。第一次下载共约 46 MB，之后使用本地缓存。无法联网时按 [数据说明](data/README.md) 手动放置文件，程序不会生成虚假训练数据。

## 3. 在 VS Code 中点击运行

当前工作区已有可用环境，日常演示无需输入终端命令：

1. 在 VS Code 中打开整个 `2024-deeplearning` 文件夹，再打开要运行的 `.py` 文件。
2. 工作区默认解释器已配置为 `/opt/conda/bin/python`。如果 VS Code 之前记住了其他解释器，点击右下角的 Python 版本，选择此路径；列表中没有时选择“输入解释器路径”。也可按 `Ctrl+Shift+P`，选择 **Python: Select Interpreter**，然后输入此路径。只需选择一次。
3. 点击编辑器右上角的三角按钮 **运行 Python 文件 / Run Python File**。下方终端自动显示输出，无需手工输入命令。
4. 等当前文件运行结束，再打开下一个文件点击运行。文件路径相对于代码位置确定，运行时无需手工切换目录。

完整演示顺序：

| 顺序 | 打开的文件 | 点击运行后的行为 |
|---|---|---|
| 1 | [prepare_data.py](prepare_data.py) | 准备数据，展示统计、编码和批次形状 |
| 2 | [check_lstm.py](check_lstm.py) | 检查 LSTM 结构及单批次前向/反向/更新 |
| 3 | [train.py](train.py) | 默认训练 LSTM，保存最佳验证模型与曲线 |
| 4 | [check_rnn.py](check_rnn.py) | 检查 RNN 结构及单批次前向/反向/更新 |
| 5 | [train_rnn.py](train_rnn.py) | 使用相同超参数训练 RNN |
| 6 | [evaluate.py](evaluate.py) | 加载两模型，完成同一测试集的最终评价 |
| 7 | [compare.py](compare.py) | 生成比较表、指标柱状图和训练曲线 |

两种模型的检查在正式训练中也会自动执行，时间紧时可省略第 2、4 步。`models.py` 和 `utils.py` 是共用模块，讲解时打开阅读即可，不属于演示运行步骤。`test_pipeline.py` 可单独点击运行功能检查。

当前已验证 Python 3.11.9、PyTorch 2.4.0、NumPy 2.4.6、Matplotlib 3.11.2；无需升级这些库。仓库上一级 `.venv` 目前没有所需依赖，请在 VS Code 中选择 `/opt/conda/bin/python`。

下面的安装命令仅供尚无环境的新电脑使用；当前工作区直接按上表点击运行即可。新电脑推荐 Python 3.10–3.12。Linux 上可执行：

```bash
cd /root/2024-deeplearning/lab03_nlp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Windows PowerShell（先进入实际项目目录）：

```powershell
cd C:\2024-deeplearning\lab03_nlp
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

macOS 在自己的项目目录执行 `python3 -m venv .venv`、`source .venv/bin/activate`，再执行 `python -m pip install -r requirements.txt` 即可。已有满足要求的环境只需按需执行最后的依赖安装命令，不加 `--upgrade`。程序始终使用 CPU。

## 4. 第 5 次实验课：准备数据和训练 LSTM

选择好解释器后，依次打开 `prepare_data.py`、`check_lstm.py`、`train.py`，分别点击右上角运行。

`prepare_data.py` 自动下载并打印类别统计、字符表大小、真实字符编码以及批次形状。默认应看到输入 `[32,256]`、标签 `[32]`、有效长度 `[32]`，字符表 3883 项。编码示例保存在 `data/processed/encoding_example.json`。

`check_lstm.py` 显示结构、参数量和实际张量形状，并在独立模型副本上验证一个批次的前向、反向和 Adam 更新。正式训练也自动做此检查，因此单独检查可省略。

正式训练默认 20 Epoch，逐轮输出训练 Loss、验证 Accuracy 和训练时间，使用验证 Accuracy 最大的轮次保存 `outputs/lstm/best.pt`，相同准确率保留较早轮次。检查在独立模型副本和随机数上下文中完成，不影响正式训练的权重、Dropout 随机序列或批次顺序。此时不要执行最终测试来选参数。

## 5. 第 6 次实验课：训练 RNN、最终测试和比较

沿用第 5 次课的数据和参数，依次打开 `check_rnn.py`、`train_rnn.py`、`evaluate.py`、`compare.py`，分别点击右上角运行。

`evaluate.py` 要求两个模型都训练完，并检查数据标识、字符表和训练条件一致，加载各自的最佳验证模型，在同一个测试集上评价。`compare.py` 汇总真实指标；条件不同或结果过期时会报错，避免把旧结果混入比较。

比较时间是 **所有 Epoch 训练批次耗时的总和**，包含批次加载、前向、反向和参数更新，不含下载、环境安装、单批次检查、验证、保存、绘图与测试。在同一台机器上顺序运行两模型，保持线程数与系统负载尽量一致。时间会随硬件和负载变化。

## 6. 默认参数与可选命令

点击运行时，训练超参数统一取自 `train.py` 顶部，包括轮数、批次大小、Embedding/隐藏维度、学习率、`POOLING`、`DROPOUT`、`WEIGHT_DECAY`、`GRAD_CLIP`、种子和线程数。修改并保存一次，LSTM 和 RNN 都使用同一设置。数据抽样数量、最大长度 `MAX_LENGTH`、种子和离线开关在 `prepare_data.py` 顶部设置；修改长度后须重新准备数据并训练两模型。

| 项目 | 默认值 |
|---|---|
| 最大长度 / Embedding / 隐藏维度 | 256 / 64 / 64 |
| 层数 / 方向 | 1 / 单向 |
| 隐藏表示汇总 | 最后有效状态（`POOLING="last"`） |
| Batch Size / Epoch | 32 / 20 |
| 优化器 / 学习率 / 损失 | Adam / 0.002 / CrossEntropyLoss |
| Dropout / 权重衰减 / 梯度裁剪范数 | 0.2 / 0.0001 / 1.0 |
| 种子 / 设备 / CPU 线程数 | 42 / CPU / 2 |

如预先指定其他训练轮数，例如 5 Epoch，在最终测试前将 `train.py` 顶部的 `EPOCHS = 20` 改为 `EPOCHS = 5`，保存后依次点击运行 `train.py`、`train_rnn.py`、`evaluate.py`、`compare.py`。轮数增加不保证验证或测试准确率提升。

原有命令行方式仍可使用，命令行参数优先于文件顶部的默认值。等价的 5 Epoch 命令为（先进入 `lab03_nlp`，并使用已安装依赖的 Python）：

```bash
python train.py --model lstm --epochs 5
python train.py --model rnn --epochs 5
python evaluate.py
python compare.py
```

`--batch-size`、`--embedding-dim`、`--hidden-dim`、`--lr`、`--pooling`、`--dropout`、`--weight-decay`、`--grad-clip`、`--seed` 和 `--threads` 同理。`--grad-clip 0` 可关闭梯度裁剪。相同种子保证两模型使用相同的初始 Embedding 和训练样本顺序；不同循环层的初始化参数形状不同。跨 PyTorch 版本不保证结果逐位一致。

教师快速排错可使用 300 条真实样本、1 Epoch，并将结果放入独立目录（指标仅用于冒烟检查，不作为基础实验成绩）：

```bash
python prepare_data.py --samples 300 --data-dir data/smoke
python train.py --model lstm --epochs 1 --data-dir data/smoke --output-dir outputs/smoke
python train.py --model rnn --epochs 1 --data-dir data/smoke --output-dir outputs/smoke
python evaluate.py --data-dir data/smoke --output-dir outputs/smoke
python compare.py --output-dir outputs/smoke
```

默认不需这些选项。重复训练同一种模型会覆盖其输出；如需保留旧结果，先复制整个输出目录或用新的 `--output-dir`。重新准备不同数据后必须重新训练两模型。不要根据测试结果反复调参。

## 7. 查看输出并完成报告

| 路径 | 内容 |
|---|---|
| `data/processed/manifest.json` | 来源、版本、清洗统计、类别数量、截断数、数据 ID 和校验值 |
| `data/processed/{train,val,test}.json` | 保留原文、星级、二分类标签、ID 和截断后有效长度的样本 |
| `data/processed/vocab.json` | 仅从训练集建立的字符表 |
| `outputs/{lstm,rnn}/best.pt` | 最佳验证模型，含权重、字符表、配置和最佳轮次 |
| `outputs/{lstm,rnn}/history.csv` | 每 Epoch 的训练 Loss、验证 Loss/Accuracy、训练时间 |
| `outputs/{lstm,rnn}/summary.json` | 参数量、最佳轮次、累计训练时间和环境 |
| `outputs/{lstm,rnn}/training_curves.png` | 各自的训练 Loss / 验证 Accuracy 曲线 |
| `outputs/{lstm,rnn}/test_metrics.json` | 最终测试指标 |
| `outputs/{lstm,rnn}/test_predictions.csv` | 全部真实测试文本及预测 |
| `outputs/misclassified.csv` | 总共至多 3 条不重复误分类，含原文、可见前缀和真假标签 |
| `outputs/comparison/comparison.csv` / `.md` | 两模型比较表 |
| `outputs/comparison/model_comparison.png` | Accuracy、参数量、时间柱状图 |
| `outputs/comparison/training_comparison.png` | 两模型训练/验证曲线叠加比较 |
| `outputs/last_state_experiment/` | 本次最后有效状态实验的方案、验证记录及版本对比 |
| `outputs/baseline_256_mean/` | 上一版平均池化的代码、处理数据和完整输出归档 |
| `outputs/optimization/validation_{plan,trials,selection}.json` | 历史 64→256 字优化方案、验证记录与选择依据 |
| `outputs/optimization/before_after.md` / `.csv` | 历史 64→256 字优化的实际结果 |
| `outputs/baseline_64chars/` | 最初 64 字版的代码、处理数据与完整输出归档 |

误分类按固定顺序选取，优先 LSTM、不足三条时由 RNN 补足；没有错误时只有 CSV 表头。可在各模型的完整预测文件中查看更多。图中轴标签使用英文以免普通环境缺少中文字体，中文含义见指导书。

使用 [实验报告模板](docs/实验报告模板.md) 填写实际指标和分析。LSTM 不一定更准确，同隐藏维度也不等于同参数量。一次实验不能证明任何模型普遍更优或长期依赖能力更强。完成基础任务即可获得满分，不要求新增模型或多种子实验。

实验记录目录和 `outputs/verification.json` 不会在每次点击运行后自动更新；日常演示应以最新 `outputs/{lstm,rnn}/` 和 `outputs/comparison/` 为准。代码支持 `last` 与 `mean` 两种表示方式，旧权重仍可通过 `model_from_config` 加载。对照平均池化时，先确定相同的训练预算，给两模型都设置 `--pooling mean`，并使用单独的 `--output-dir` 保留结果。

## 8. 功能检查与排错

```bash
python -m compileall -q prepare_data.py models.py train.py train_rnn.py check_lstm.py check_rnn.py evaluate.py compare.py utils.py test_pipeline.py
python -m unittest -v test_pipeline
```

也可直接打开 `test_pipeline.py`，点击右上角运行全部功能检查。功能检查里的临时文本只验证程序行为，不输出或冒充真实实验指标。真实端到端验证记录见 [验证记录](docs/验证记录.md)。

- `No module named torch`：点击 VS Code 右下角的 Python 版本，选择 `/opt/conda/bin/python`。
- 下载失败：看 `data/README.md` 的固定直链、文件名和格式，手动放好文件后，将 `prepare_data.py` 顶部的 `OFFLINE` 改为 `True` 再点击运行。
- 找不到模型：先顺序完成两模型训练，再执行最终测试。
- 比较条件不同/文件已改变：确认两个训练命令一致，必要时重新准备数据并训练。不要复用其他数据上的权重。
- CPU 较慢：关闭占用资源的程序；若将 `train.py` 顶部的 `THREADS` 改为 `1`，两个模型会共用此设置。基础任务无需 GPU。
