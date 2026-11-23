# 实验三 · CIFAR-10 图像分类

本仓库的默认模型是报告中的**完整方案**。它使用自设计的四阶段 CifarCNN，从头训练，采用 RandAugment 和余弦学习率调度。原实验在第 98 轮达到 **93.78%** 验证准确率，官方测试集准确率为 **93.43%（9,343/10,000）**。

| 任务 | 运行入口 |
| --- | --- |
| 完整模型的训练、测试和绘图 | [scripts/run_final_model.py](scripts/run_final_model.py) |
| 固定学习率、去除 RandAugment 等 32 组额外实验，以及验证集扰动对照 | [scripts/run_additional_experiments.py](scripts/run_additional_experiments.py) |

## 环境与安装

| 项目 | 原始实验环境 |
| --- | --- |
| 平台 | **Linux x86_64，NVIDIA CUDA GPU** |
| 显卡 | **NVIDIA GeForce RTX 3090，24 GB 显存** |
| Python | **3.11.15** |
| PyTorch / torchvision | **2.10.0 / 0.25.0** |
| PyTorch CUDA 构建版本 | 12.8 |
| NVIDIA 驱动 | 550.90.07 |
| NumPy / Matplotlib / Pillow | 1.26.4 / 3.7.5 / 12.1.1 |

完整的平台、软件版本和运行命令保存在 `environment.json`，见 [正式测试归档](results/final_test/)。

原始实验环境的部分依赖存在版本约束冲突。在仓库根目录创建 `project3` Conda 环境，使用 `--no-deps` 跳过依赖解析并安装清单记录的版本。清单含 CUDA 和 Triton 等平台相关包，按原实验的 Linux 环境安装。训练和测试需要有可用的 NVIDIA CUDA GPU，使用现有结果绘图可以在 CPU 上完成。依赖版本见 [requirements.txt](requirements.txt)。

```shell
conda create -n project3 python=3.11.15 -y
conda activate project3
python -m pip install --no-deps -r requirements.txt
```

## 目录结构

```text
.
├── config/final.json                  # 默认完整模型配置
├── scripts/
│   ├── run_final_model.py             # 完整模型
│   └── run_additional_experiments.py  # 额外实验和扰动对照
├── run.py                            # 共享的训练、评估和绘图接口
├── models/                           # 主干、分类头和结构变体
├── data.py                           # 数据下载、校验、划分与增强
├── experiment.py                     # GPU 预检、训练、评估与预测导出
├── recipes.py                        # 额外实验的参数定义与配置校验
├── visualization.py                  # 绘制学习曲线、混淆矩阵、错例和类激活图
├── dataset/                          # 本地 CIFAR-10 数据
├── results/                          # 新运行结果和原实验归档
│   ├── complete_model/               # 新完整模型训练结果
│   ├── additional_experiments/       # 新额外实验结果
│   └── final_test/                   # 完整模型正式测试归档
└── figures/                          # 图表
```

## 数据下载与放置

使用 CIFAR-10 Python 版。手动下载压缩包后，将它放到仓库根目录的 `dataset/cifar-10-python.tar.gz`。运行下面的命令会优先校验并解压本地压缩包。官方压缩包为 [cifar-10-python.tar.gz](https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz)。

```shell
python scripts/run_final_model.py download --data-dir dataset
```

自行解压时，将 `cifar-10-batches-py/` 直接放在 `dataset/` 下。程序对五个训练批次、测试批次和类别元数据校验 MD5。已完整解压时，训练命令可以加 `--no-download`。使用其他数据目录时，让 `--data-dir` 指向 `cifar-10-batches-py/` 的父目录。

```text
project3/
└── dataset/
    ├── cifar-10-python.tar.gz       # 解压后可省略
    └── cifar-10-batches-py/
        ├── batches.meta
        ├── data_batch_1
        ├── data_batch_2
        ├── data_batch_3
        ├── data_batch_4
        ├── data_batch_5
        └── test_batch
```

torchvision 直接读取原始批次的 `data` 和 `labels`。模型输入是形状为 `[B, 3, 32, 32]` 的 RGB 张量，输出为 `[B, 10]` 的 logits。类别编号从 0 开始，顺序为 airplane、automobile、bird、cat、deer、dog、frog、horse、ship、truck。

## 完整模型

四阶段主干采用 `[64, 128, 256, 384]` 通道和 `[2, 2, 3, 3]` 层的 `3×3 Conv + BatchNorm + ReLU`，前三阶段末尾进行 `2×2` 最大池化，最后保留 `384×4×4` 特征图。分类头依次使用 `1×1` 卷积、全局平均池化、Dropout 和线性分类，共有 **5,328,586** 个可训练参数。完整训练参数见 [config/final.json](config/final.json)。

| 项目 | 设置 |
| --- | --- |
| 数据划分 | 官方训练集按类别分层划分为 45,000 张训练图和 5,000 张验证图，每类分别有 4,500 和 500 张 |
| 测试集 | 官方 10,000 张，每类 1,000 张 |
| 随机种子 | 数据划分和训练均使用 42，运行目录保存精确划分索引 |
| 训练增强 | 填充 4 像素后随机裁剪至 32×32，以 0.5 概率水平翻转，再应用 RandAugment |
| RandAugment | `num_ops=2`、`magnitude=9`、双线性插值、填充值 128 |
| 归一化 | 三个通道的 mean 和 std 均为 0.5 |
| 训练预算 | 100 轮、batch size 128、默认 4 个 DataLoader workers |
| 优化 | 交叉熵、Adam、权重衰减 `1e-4`、Dropout 0.3 |
| 学习率 | 余弦调度，`T_max=99`，从 `1e-3` 降至 `1e-5` |
| 模型选择 | 选验证准确率最高的轮次，同分时取最早轮次；正式训练跑完 100 轮 |

### 训练与评估

下载并解压数据后，在仓库根目录设置 Python 哈希种子，再运行 GPU 预检和训练。普通入口 `python run.py train` 也默认使用完整方案。

```shell
export PYTHONHASHSEED=42
python scripts/run_final_model.py check --gpu 0
python scripts/run_final_model.py train --gpu 0 --data-dir dataset --no-download
```

训练会打印本次运行目录，保存最佳权重，并自动评估无增强训练集、验证集和官方测试集。新运行结果写入 `results/complete_model/`，图表写入 `figures/complete_model/`。正式评估使用真实硬标签交叉熵，在评估模式下做单视图推理。

将命令中的示例路径替换为实际训练目录，可再次评估该次运行保存的 `best.pt`。

```shell
python scripts/run_final_model.py test --run-dir 'results/complete_model/实际训练目录' --gpu 0 --data-dir dataset --no-download
```

复现报告中的正式测试时，先按[复现文件](#复现文件)中的命令找到完整模型训练目录，把原始第 98 轮的 `best.pt` 放回该目录，再运行下列命令。脚本校验权重 SHA-256、历史记录和验证准确率，将本次输出写入 `results/complete_model/official_test/` 和 `figures/complete_model/official_test/`。报告里的 **93.43%** 对应这份原始权重。独立训练的模型使用上面的 `--run-dir` 命令评估。

```shell
python scripts/run_final_model.py test --original --data-dir dataset
```

只重绘报告中的图表时，可以直接读取已保存的正式测试结果和绘图缓存。

```shell
python scripts/run_final_model.py plot --original
```

### 正式测试结果

| 指标 | 数值 |
| --- | ---: |
| 最佳轮次 | 98 |
| 验证准确率 | 93.78% |
| 测试准确率 | **93.43%** |
| 测试交叉熵 | 0.257862 |
| 测试 Macro-F1 | 0.934106 |
| 测试 ECE | 4.0287% |
| 错例数 | 657 |

测试指标保存在 `test_metrics.json`，见 [正式测试归档](results/final_test/)。学习曲线、混淆矩阵、逐类指标、置信度诊断、错例和类激活图共六类，见 [正式测试图表归档](figures/final_test/)。

最终预测文件为正式测试归档中的 `test_predictions.csv`。文件采用 UTF-8 编码，每张测试图一行，共 10,000 行数据。样本索引对应官方测试集的 0–9,999，类别标签是 0–9 的整数。置信度取预测类别的 softmax 概率，`top2_margin` 为前两名概率之差。

```text
sample_index,true_label,true_class,predicted_label,predicted_class,confidence,correct,true_class_probability,cross_entropy,top2_margin
```

同目录的 `test_outputs.npz` 保留样本索引、真实标签和形状为 `[10000, 10]` 的 logits。逐类准确率和混淆矩阵见 `test_class_accuracy.csv` 与 `test_confusion_matrix.csv`。

## 额外实验

以**去除 RandAugment** 为共同参照。这组保留完整模型的网络、裁剪与翻转、Adam 和余弦调度，验证准确率为 **93.12%**。固定学习率对照将调度改为恒定 `1e-3`，验证准确率为 **90.78%**。完整方案和额外实验共 33 组，使用同一划分与随机种子。表内为原实验各组最佳轮次的**验证准确率**。

| 类别 | 方案 | 命令参数 | 验证准确率 |
| --- | --- | --- | ---: |
| 参照 | 去除 RandAugment | `without_randaugment` | 93.12% |
| 数据增强 | 随机擦除 0.10 | `random_erasing_010` | 93.12% |
| 数据增强 | 随机擦除 0.25 | `random_erasing_025` | 93.04% |
| 数据增强 | 随机擦除 0.50 | `random_erasing_050` | 93.54% |
| 数据增强 | Mixup | `mixup` | 93.38% |
| 数据增强 | CutMix | `cutmix` | 93.52% |
| 数据增强 | 颜色抖动 | `color_jitter` | 92.94% |
| 结构与注意力 | 多尺度融合 | `multiscale_fusion` | 92.56% |
| 结构与注意力 | 空间注意力 | `spatial_attention` | 92.52% |
| 结构与注意力 | 多尺度融合 + 空间注意力 | `multiscale_fusion_spatial_attention` | 92.66% |
| 结构与注意力 | 分类头 SE | `head_se` | 92.96% |
| 结构与注意力 | 分类头 SE + 空间注意力 | `head_se_spatial_attention` | 92.62% |
| 结构与注意力 | 第二阶段 SE | `stage2_se` | 93.26% |
| 结构与注意力 | 全局最大池化 | `global_max_pooling` | 93.12% |
| 结构与注意力 | 全局平均与最大池化各半 | `global_average_max_pooling` | 92.94% |
| 结构与注意力 | 主干改用平均池化 | `average_downsampling` | 92.68% |
| 结构与注意力 | 阶段残差连接 | `stage_residual` | 93.16% |
| 训练与优化 | 固定学习率 | `fixed_learning_rate` | 90.78% |
| 训练与优化 | Dropout 0 | `dropout_0` | 92.94% |
| 训练与优化 | Dropout 0.5 | `dropout_05` | 92.70% |
| 训练与优化 | AdamW（权重衰减 `1e-4`） | `adamw_1e4` | 92.72% |
| 训练与优化 | AdamW（权重衰减 `1e-2`） | `adamw_1e2` | 92.58% |
| 训练与优化 | 动量 SGD | `momentum_sgd` | 93.26% |
| 训练与优化 | 按验证损失降学习率 | `validation_loss_scheduler` | 92.92% |
| 损失函数 | 标签平滑 0.02 | `label_smoothing_002` | 93.50% |
| 损失函数 | 标签平滑 0.05 | `label_smoothing_005` | 92.50% |
| 损失函数 | 标签平滑 0.10 | `label_smoothing_010` | 92.80% |
| 损失函数 | 标签平滑 0.20 | `label_smoothing_020` | 92.46% |
| 组合消融 | 擦除 + SE | `random_erasing_head_se` | 93.42% |
| 组合消融 | 擦除 + 平滑 | `random_erasing_label_smoothing` | 93.62% |
| 组合消融 | SE + 平滑 | `head_se_label_smoothing` | 92.80% |
| 组合消融 | 擦除 + SE + 平滑 | `random_erasing_head_se_label_smoothing` | 93.40% |

组合消融使用擦除概率 0.25、分类头 SE 和标签平滑 0.05。随机擦除组统一采用面积比例 `[0.02, 0.15]`、长宽比 `[0.3, 3.3]`，填充值为归一化后的 0。Mixup 使用 `alpha=0.2`，每个训练批次启用。CutMix 使用 `alpha=1.0`，启用概率为 0.50。动量 SGD 使用 0.9 动量，学习率从 0.05 余弦降至 0.0005。按验证损失降学习率组使用衰减因子 0.5、等待 5 轮、绝对阈值 `1e-4` 和最低学习率 `1e-5`。详细配置见 [recipes.py](recipes.py)。

### 运行额外实验

使用 `--experiments` 指定一个或多个方案。省略该参数会依次训练全部 32 组。`--dry-run` 可先查看配置和所选方案。新结果按名称写入 `results/additional_experiments/<批次>/<方案名>/`，同批次的 `metrics.csv` 和 `summary.json` 汇总指标。已有批次可用 `--resume <批次目录>` 恢复，续训需要对应的 `last.pt`。

```shell
python scripts/run_additional_experiments.py train --experiments fixed_learning_rate without_randaugment random_erasing_050 --gpu 0 --data-dir dataset --no-download
```

额外实验沿用原有评估范围。固定学习率、去除 RandAugment 及三种多尺度或空间注意力方案评估训练集、验证集和测试集，并生成图表。随机擦除 0.25、标签平滑 0.10 和分类头 SE 也评估这三个数据集。其余方案评估训练集和验证集，表中的方案比较只使用验证数据。

### 扰动对照与早停模拟

扰动对照比较完整模型、去除 RandAugment 和随机擦除 0.50。原实验用同一批 5,000 张验证图，分别施加左右平移 2 像素、亮度乘 0.7 或 1.3、4×4 或 8×8 遮挡。模型间共享遮挡位置。亮度变换在归一化前完成，平移和遮挡以归一化后的 0 填充。原实验的 8×8 遮挡下，完整模型和随机擦除 0.50 的准确率分别为 88.42% 和 91.66%。

恢复三组原始权重后运行下列命令。若权重放在其他目录，使用 `--complete-model-run`、`--without-randaugment-run` 和 `--random-erasing-run` 指定目录。新运行写入 `results/additional_experiments/robustness/`。

```shell
python scripts/run_additional_experiments.py robustness --gpu 0 --data-dir dataset
```

报告的早停模拟读取各次训练的 `history.csv`。验证准确率连续 10 轮未刷新时，记录当时最佳轮次。原完整方案在第 94 轮触发，保留第 84 轮的 93.68%。33 组实验中 19 组提前停止，18 组错过完整训练峰值。正式实验均完成预设训练轮次。

### 原实验结果

| 内容 | 文件 |
| --- | --- |
| 固定学习率 | [训练和评估记录](results/basic/cifarcnn_20261118T160642Z_93925a89) |
| 去除 RandAugment、多尺度融合和空间注意力 | [comparison.csv](results/ablations/suite_20261119T072348Z_e0a7c9de/comparison.csv) |
| 随机擦除 0.25、标签平滑 0.10、分类头 SE | `suite_20261120T122731Z_6ea426b2` 批次的 `comparison.csv`，从 [原实验结果](results/)查找 |
| 其余对照及完整模型的训练 | [metrics.csv](results/extended_experiments/suite_20261120T170637Z_21e15ceb/metrics.csv) |
| 扰动对照 | [扰动对照归档](results/robustness/)内的 `accuracy_drop.csv` 和 `per_class_drop.csv` |

## 复现文件

`.gitignore` 内的内容默认不上传。需要额外准备或从原实验机器备份的文件如下。

| 本地文件 | 用途 |
| --- | --- |
| `dataset/` | CIFAR-10 原始数据和自动生成的 `split_seed42.npz`。原始数据按上文下载，各次运行的精确划分另存于 `split_indices.npz` |
| `best.pt`、其他 `*.pt` 或 `*.pth` | 评估、扰动对照及重新计算 CAM 所需的模型权重 |
| `last.pt` | 续训所需的最近完整轮次状态，含模型、优化器、调度器和随机数状态 |
| `*.log` | 训练或评估的日志，排查阶段性失败时可用 |
| `.venv/`、`venv/`、`.env`、`.env.*` | 本机环境及变量配置，依赖使用 `requirements.txt` 重建 |

`tmp/`、Python 缓存、编辑器配置和系统临时文件由本地工具管理。

原始正式测试需要完整模型的 `best.pt`。扰动对照还需要去除 RandAugment 和随机擦除 0.50 两组的 `best.pt`。在仓库根目录运行以下命令，按这个顺序打印三个训练目录，把各自的权重放回对应目录。

```shell
python -c "from scripts import run_final_model as m; print(m.COMPLETE_MODEL_RUN, m.WITHOUT_RANDAUGMENT_RUN, m.RANDOM_ERASING_050_RUN, sep='\n')"
```

原始完整模型权重的 SHA-256 为 `066dc8b8de172f5ee1bc08cebee6cdf16bc41fea4b16c189b2b63c38df32f1d2`。

如需核验复现结果，查看 `status.json` 的完成状态，核对 `history.csv` 的最佳轮次，以及验证和测试指标关联的 `best_checkpoint_sha256`。可通过 `artifact_sha256` 验证预测文件与缓存的完整性。原始配置中的绝对路径记录当时的机器位置，绘图时可使用 `--figures-dir` 指定本机目录。

所有配置采用单一随机种子。不同 GPU、驱动或库版本可能带来数值差异。原实验的准确率、Macro-F1 和 ECE 在 JSON/CSV 中存为小数。ECE 使用 10 个等宽置信度区间，鲁棒性下降幅度以百分点计。完整模型的 CAM 基于 4×4 特征图，每张热图独立归一化，适合比较响应的空间位置。
