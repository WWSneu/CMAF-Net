# CMAF-Net：面向中文虚假评论检测的细粒度评价单元与元数据融合模型

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

CMAF-Net 是一个用于中文虚假餐厅评论检测的 PyTorch 实现。模型融合了评论文本语义、细粒度评价单元、结构化元数据以及随机森林元数据专家分数，用于完成真假评论二分类任务。

本仓库主要面向论文/课程报告复现实验与公开展示，包含模型源码、训练入口、实验结果摘要、消融结果和模型架构图。

## 方法概述

虚假评论的判别线索通常不只存在于整条评论的整体情感中，而是隐藏在局部表达模式里。例如，一条评论可能同时包含真实消费过程、正负混合体验、夸张情绪词、模板化推荐语，或评分与文本证据不一致等现象。CMAF-Net 通过多分支结构对这些信息进行联合建模：

1. **全局文本分支**：使用中文 RoBERTa 编码整条评论，获得整体语义表示。
2. **Claim 细粒度评价单元分支**：将评论切分为多个分句，通过可学习查询向量抽取局部评价单元。
3. **元数据辅助分支**：利用评分、互动、图片、时间、文本统计等结构化特征学习高层元数据表示。
4. **随机森林专家分支**：基于元数据生成 out-of-fold 专家概率分数，补充传统机器学习视角。
5. **融合分类器**：将全局语义、局部评价单元、元数据表示和专家分数融合，输出评论真假类别。

![CMAF-Net 模型架构](docs/figures/architecture.png)

## 仓库结构

```text
CMAF-Net/
├── main_train.py              # 训练入口：交叉验证、阈值搜索、曲线保存、消融实验
├── requirements.txt           # Python 依赖
├── src/
│   ├── data_prep.py           # 数据读取、分句、元数据特征构建
│   ├── dataset.py             # Dataset 与 collator
│   ├── experiment.py          # 单折训练、十折实验、Strict-CV、消融实验
│   ├── meta_sampling.py       # NearMiss、标准化、RF OOF 专家分数
│   ├── model.py               # CMAF-Net 模型结构与 Claim 损失
│   ├── training_utils.py      # 指标、阈值搜索、随机种子、早停工具
│   └── visualize.py           # Claim 可解释性可视化工具
├── data/
│   └── README.md              # 数据格式说明，原始数据不随仓库发布
├── docs/
│   └── figures/               # 模型架构图
└── results/                   # 轻量级复现实验结果与曲线
```

## 环境安装

建议使用 Conda 创建独立环境：

```bash
conda create -n cmaf python=3.10 -y
conda activate cmaf
pip install -r requirements.txt
```

默认文本主干模型为：

```text
hfl/chinese-roberta-wwm-ext
```

首次运行时会从 Hugging Face 下载模型权重。

## 数据格式

请将训练数据放置为：

```text
data/merged_data.csv
```

CSV 至少需要包含以下字段：

| 字段 | 含义 |
|---|---|
| `text` | 评论文本 |
| `label` | 二分类标签，`1` 表示虚假评论，`0` 表示真实评论 |

完整元数据字段请参考 [data/README.md](data/README.md)。如果缺少部分元数据列，代码会自动用 0 填充；但若希望复现完整模型效果，建议使用包含评分、互动、图片、时间等字段的完整数据。

由于平台评论数据可能涉及再分发限制，仓库不包含原始数据文件。

## 训练主实验

运行 paper-aligned 十折交叉验证实验：

```bash
python -u main_train.py \
  --csv_path data/merged_data.csv \
  --model_name hfl/chinese-roberta-wwm-ext \
  --n_splits 10 \
  --random_state 42 \
  --cuda_visible_devices 0 \
  --output_dir outputs
```

服务器后台运行示例：

```bash
nohup python -u main_train.py \
  --csv_path data/merged_data.csv \
  --model_name hfl/chinese-roberta-wwm-ext \
  --n_splits 10 \
  --random_state 42 \
  --cuda_visible_devices 0 \
  --output_dir outputs > train.log 2>&1 &
```

主实验会输出：

- `outputs/cv_results.csv`
- `outputs/balanced_data.csv`
- `outputs/oof_roc.png`
- `outputs/oof_pr.png`

## Strict-CV 评估

默认 paper-aligned 设置会先做全局 NearMiss 平衡，再进行交叉验证。仓库同时提供更严格的 Strict-CV 协议：每个 fold 内只对训练集做 NearMiss，验证集保持原始分布。

```bash
python -u main_train.py \
  --csv_path data/merged_data.csv \
  --model_name hfl/chinese-roberta-wwm-ext \
  --n_splits 10 \
  --random_state 42 \
  --cuda_visible_devices 0 \
  --output_dir outputs \
  --strict_cv
```

Strict-CV 输出文件：

- `outputs/strict_cv_results.csv`
- `outputs/strict_oof_roc.png`
- `outputs/strict_oof_pr.png`

## 消融实验

运行完整模型和四组核心消融：

```bash
python -u main_train.py \
  --csv_path data/merged_data.csv \
  --model_name hfl/chinese-roberta-wwm-ext \
  --n_splits 10 \
  --random_state 42 \
  --cuda_visible_devices 0 \
  --output_dir outputs \
  --run_ablation \
  --ablation_quick
```

只运行消融实验：

```bash
python -u main_train.py \
  --csv_path data/merged_data.csv \
  --model_name hfl/chinese-roberta-wwm-ext \
  --n_splits 10 \
  --random_state 42 \
  --cuda_visible_devices 0 \
  --output_dir outputs \
  --ablation_only \
  --ablation_quick
```

消融结果保存为：

```text
outputs/ablation_results.csv
```

## 已归档实验结果

用于报告与复现检查的轻量级结果文件已归档在 [results/](results/)。

### 十折交叉验证结果

| 指标 | 均值 | 标准差 |
|---|---:|---:|
| Accuracy | 0.9272 | 0.0097 |
| Precision | 0.9340 | 0.0159 |
| Recall | 0.9199 | 0.0222 |
| F1 | 0.9266 | 0.0102 |
| AUC | 0.9791 | 0.0050 |
| AP | 0.9796 | 0.0060 |

### 核心消融结果

| 模型变体 | Accuracy | Precision | Recall | F1 | AUC | AP |
|---|---:|---:|---:|---:|---:|---:|
| FULL | 0.9279 | 0.9333 | 0.9224 | 0.9275 | 0.9800 | 0.9808 |
| NO_RF | 0.9293 | 0.9354 | 0.9229 | 0.9289 | 0.9805 | 0.9816 |
| NO_CLAIM | 0.9295 | 0.9452 | 0.9125 | 0.9282 | 0.9814 | 0.9825 |
| NO_META | 0.8753 | 0.8835 | 0.8660 | 0.8742 | 0.9482 | 0.9495 |

## 可复现性说明

- 代码通过 `set_global_seed` 设置 Python、NumPy 与 PyTorch 随机种子。
- CUDA 可用时启用 deterministic 设置。
- 随机森林元数据专家分数采用 out-of-fold 方式生成，降低信息泄漏风险。
- 原始数据、平衡后的中间数据、模型权重、日志和训练输出不纳入版本控制。

## 引用

如果你使用本仓库，请参考 [CITATION.cff](CITATION.cff) 中的元数据进行引用。

## 许可证

本项目采用 MIT License，详见 [LICENSE](LICENSE)。
