# claim_train_nohup

把 `07_claim_pro_paper.ipynb` 拆分为可在服务器上用 `nohup` 运行的 Python 工程。

## 目录结构

- `main_train.py`: 训练入口（含 OOF 指标、阈值搜索、曲线保存）
- `src/data_prep.py`: 数据读取、分句、元特征构建
- `src/model.py`: 模型与损失
- `src/dataset.py`: Dataset 与 collator
- `src/meta_sampling.py`: NearMiss、标准化、RF OOF 分数
- `src/training_utils.py`: 指标、early stop、优化器
- `src/experiment.py`: 单折训练与 KFold 主流程
- `src/visualize.py`: claim 可视化工具

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
cd /Users/wensongwang/code/project/claim_train_nohup
nohup python -u main_train.py \
  --csv_path data/merged_data.csv \
  --model_name hfl/chinese-roberta-wwm-ext \
  --n_splits 10 \
  --random_state 42 \
  --cuda_visible_devices 6 \
  --output_dir outputs > train.log 2>&1 &
```

查看日志：

```bash
tail -f train.log
```

## 消融实验

主实验后追加消融：

```bash
nohup python -u main_train.py \
  --csv_path data/merged_data.csv \
  --model_name hfl/chinese-roberta-wwm-ext \
  --n_splits 10 \
  --random_state 42 \
  --cuda_visible_devices 6 \
  --output_dir outputs \
  --run_ablation > train_ablation.log 2>&1 &
```

只跑 4 组核心消融（FULL/NO_CLAIM/NO_META/NO_RF）：

```bash
nohup python -u main_train.py \
  --csv_path data/merged_data.csv \
  --model_name hfl/chinese-roberta-wwm-ext \
  --n_splits 10 \
  --random_state 42 \
  --cuda_visible_devices 6 \
  --output_dir outputs \
  --run_ablation \
  --ablation_quick > train_ablation_quick.log 2>&1 &
```

只跑消融，不跑主实验：

```bash
nohup python -u main_train.py \
  --csv_path data/merged_data.csv \
  --model_name hfl/chinese-roberta-wwm-ext \
  --n_splits 10 \
  --random_state 42 \
  --cuda_visible_devices 6 \
  --output_dir outputs \
  --ablation_only > ablation_only.log 2>&1 &
```

结果会额外输出：`outputs/ablation_results.csv`
