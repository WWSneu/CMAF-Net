import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import Trainer, TrainingArguments
from transformers.utils.notebook import NotebookProgressCallback

from .dataset import ClaimMetaDataset, claim_meta_collator
from .meta_sampling import apply_nearmiss_by_city, build_rf_oof_scores, fit_meta_scaler
from .model import RobertaClaimMetaFusion, RobertaClaimMetaFusionAblation
from .training_utils import (
    InMemoryEarlyStoppingCallback,
    build_optimizer,
    compute_metrics,
    metrics_from_probs,
)


def run_single_fold(
    train_df,
    val_df,
    tokenizer,
    meta_cols,
    device,
    model_name="hfl/chinese-roberta-wwm-ext",
    hidden_dim=256,
    num_claims=4,
    dropout=0.1,
    diversity_lambda=0.005,
    max_length=160,
    random_state=42,
):
    x_train_meta, x_val_meta, meta_scaler = fit_meta_scaler(train_df, val_df, meta_cols)

    y_train = train_df["label"].values.astype(int)
    rf_train_scores, rf_val_scores, rf_model = build_rf_oof_scores(
        x_train_meta, y_train, x_val_meta, random_state=random_state
    )

    train_dataset = ClaimMetaDataset(
        train_df,
        tokenizer=tokenizer,
        meta_matrix=x_train_meta,
        rf_scores=rf_train_scores,
        max_length=max_length,
    )
    val_dataset = ClaimMetaDataset(
        val_df,
        tokenizer=tokenizer,
        meta_matrix=x_val_meta,
        rf_scores=rf_val_scores,
        max_length=max_length,
    )

    model = RobertaClaimMetaFusion(
        model_name=model_name,
        tokenizer=tokenizer,
        meta_dim=len(meta_cols),
        hidden_dim=hidden_dim,
        num_claims=num_claims,
        num_labels=2,
        dropout=dropout,
        temperature=0.7,
        diversity_lambda=diversity_lambda,
    ).to(device)

    for _, param in model.encoder.named_parameters():
        param.requires_grad = False

    training_args_stage1 = TrainingArguments(
        output_dir="./tmp_stage1",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=20,
        report_to="none",
        remove_unused_columns=False,
        learning_rate=1e-4,
        weight_decay=0.01,
    )

    trainer_stage1 = Trainer(
        model=model,
        args=training_args_stage1,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=claim_meta_collator,
    )

    try:
        trainer_stage1.remove_callback(NotebookProgressCallback)
    except Exception:
        pass

    print("🔥 Stage 1: 冻结 RoBERTa，只训练 claim + metadata 分支")
    trainer_stage1.train()

    for _, param in model.encoder.named_parameters():
        param.requires_grad = True

    early_stop_cb = InMemoryEarlyStoppingCallback(
        metric_name="eval_f1", patience=2, greater_is_better=True
    )

    training_args_stage2 = TrainingArguments(
        output_dir="./tmp_stage2",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to="none",
        remove_unused_columns=False,
        weight_decay=0.01,
    )

    optimizer = build_optimizer(
        model, encoder_lr=2e-5, head_lr=1e-4, weight_decay=0.01
    )

    trainer_stage2 = Trainer(
        model=model,
        args=training_args_stage2,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=claim_meta_collator,
        optimizers=(optimizer, None),
        callbacks=[early_stop_cb],
    )

    try:
        trainer_stage2.remove_callback(NotebookProgressCallback)
    except Exception:
        pass

    print("🔥 Stage 2: 解冻 RoBERTa，联合训练")
    trainer_stage2.train()

    early_stop_cb.restore_best_weights(model)

    pred_output = trainer_stage2.predict(val_dataset)
    logits = pred_output.predictions
    labels = pred_output.label_ids

    if isinstance(logits, tuple):
        logits = logits[0]

    probs = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()
    probs_pos = probs[:, 1]

    fold_metrics = metrics_from_probs(labels, probs_pos, threshold=0.5)

    artifacts = {
        "model": model,
        "meta_scaler": meta_scaler,
        "rf_model": rf_model,
        "probs_pos": probs_pos,
        "labels": labels,
        "val_df": val_df.reset_index(drop=True),
    }

    return fold_metrics, artifacts


def run_single_fold_ablation(
    train_df,
    val_df,
    tokenizer,
    meta_cols,
    device,
    model_name="hfl/chinese-roberta-wwm-ext",
    hidden_dim=256,
    num_claims=4,
    dropout=0.1,
    diversity_lambda=0.005,
    max_length=160,
    random_state=42,
    use_cls=True,
    use_claim=True,
    use_meta=True,
    use_rf=True,
):
    x_train_meta, x_val_meta, meta_scaler = fit_meta_scaler(train_df, val_df, meta_cols)
    y_train = train_df["label"].values.astype(int)

    if use_rf:
        rf_train_scores, rf_val_scores, rf_model = build_rf_oof_scores(
            x_train_meta, y_train, x_val_meta, random_state=random_state
        )
    else:
        rf_train_scores = np.zeros(len(train_df), dtype=np.float32)
        rf_val_scores = np.zeros(len(val_df), dtype=np.float32)
        rf_model = None

    train_dataset = ClaimMetaDataset(
        train_df,
        tokenizer=tokenizer,
        meta_matrix=x_train_meta,
        rf_scores=rf_train_scores,
        max_length=max_length,
    )
    val_dataset = ClaimMetaDataset(
        val_df,
        tokenizer=tokenizer,
        meta_matrix=x_val_meta,
        rf_scores=rf_val_scores,
        max_length=max_length,
    )

    model = RobertaClaimMetaFusionAblation(
        model_name=model_name,
        tokenizer=tokenizer,
        meta_dim=len(meta_cols),
        hidden_dim=hidden_dim,
        num_claims=num_claims,
        num_labels=2,
        dropout=dropout,
        temperature=0.7,
        diversity_lambda=diversity_lambda,
        use_cls=use_cls,
        use_claim=use_claim,
        use_meta=use_meta,
        use_rf=use_rf,
    ).to(device)

    for _, param in model.encoder.named_parameters():
        param.requires_grad = False

    training_args_stage1 = TrainingArguments(
        output_dir="./tmp_stage1_ablation",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=20,
        report_to="none",
        remove_unused_columns=False,
        learning_rate=1e-4,
        weight_decay=0.01,
    )

    trainer_stage1 = Trainer(
        model=model,
        args=training_args_stage1,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=claim_meta_collator,
    )

    try:
        trainer_stage1.remove_callback(NotebookProgressCallback)
    except Exception:
        pass

    print("🔥 Stage 1: 冻结 RoBERTa，训练可开启的融合分支")
    trainer_stage1.train()

    for _, param in model.encoder.named_parameters():
        param.requires_grad = True

    early_stop_cb = InMemoryEarlyStoppingCallback(
        metric_name="eval_f1", patience=2, greater_is_better=True
    )

    training_args_stage2 = TrainingArguments(
        output_dir="./tmp_stage2_ablation",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to="none",
        remove_unused_columns=False,
        weight_decay=0.01,
    )

    optimizer = build_optimizer(
        model, encoder_lr=2e-5, head_lr=1e-4, weight_decay=0.01
    )

    trainer_stage2 = Trainer(
        model=model,
        args=training_args_stage2,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=claim_meta_collator,
        optimizers=(optimizer, None),
        callbacks=[early_stop_cb],
    )

    try:
        trainer_stage2.remove_callback(NotebookProgressCallback)
    except Exception:
        pass

    print("🔥 Stage 2: 解冻 RoBERTa，联合训练")
    trainer_stage2.train()
    early_stop_cb.restore_best_weights(model)

    pred_output = trainer_stage2.predict(val_dataset)
    logits = pred_output.predictions
    labels = pred_output.label_ids

    if isinstance(logits, tuple):
        logits = logits[0]

    probs = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()
    probs_pos = probs[:, 1]

    fold_metrics = metrics_from_probs(labels, probs_pos, threshold=0.5)
    artifacts = {
        "model": model,
        "meta_scaler": meta_scaler,
        "rf_model": rf_model,
        "probs_pos": probs_pos,
        "labels": labels,
        "val_df": val_df.reset_index(drop=True),
    }

    return fold_metrics, artifacts


def run_ablation_cv_experiment(
    balanced_df,
    meta_cols,
    tokenizer,
    device,
    model_name="hfl/chinese-roberta-wwm-ext",
    random_state=42,
    n_splits=10,
    quick_mode=False,
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    full_settings = {
        "FULL": {"use_cls": True, "use_claim": True, "use_meta": True, "use_rf": True},
        "NO_CLAIM": {"use_cls": True, "use_claim": False, "use_meta": True, "use_rf": True},
        "NO_META": {"use_cls": True, "use_claim": True, "use_meta": False, "use_rf": True},
        "NO_RF": {"use_cls": True, "use_claim": True, "use_meta": True, "use_rf": False},
        "TEXT_ONLY": {"use_cls": True, "use_claim": True, "use_meta": False, "use_rf": False},
        "META_ONLY": {"use_cls": False, "use_claim": False, "use_meta": True, "use_rf": True},
    }
    quick_settings = {
        "FULL": {"use_cls": True, "use_claim": True, "use_meta": True, "use_rf": True},
        "NO_CLAIM": {"use_cls": True, "use_claim": False, "use_meta": True, "use_rf": True},
        "NO_META": {"use_cls": True, "use_claim": True, "use_meta": False, "use_rf": True},
        "NO_RF": {"use_cls": True, "use_claim": True, "use_meta": True, "use_rf": False},
    }
    ablation_settings = quick_settings if quick_mode else full_settings

    all_results = []

    for exp_name, cfg in ablation_settings.items():
        print(f"\n\n================ {exp_name} ================\n")
        fold_rows = []

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(balanced_df["text"].astype(str).values, balanced_df["label"].values),
            start=1,
        ):
            print(f"[{exp_name}] Fold {fold}/{n_splits}")

            train_df = balanced_df.iloc[train_idx].reset_index(drop=True)
            val_df = balanced_df.iloc[val_idx].reset_index(drop=True)

            fold_metrics, _ = run_single_fold_ablation(
                train_df=train_df,
                val_df=val_df,
                tokenizer=tokenizer,
                meta_cols=meta_cols,
                device=device,
                model_name=model_name,
                hidden_dim=256,
                num_claims=4,
                dropout=0.1,
                diversity_lambda=0.005,
                max_length=160,
                random_state=random_state + fold,
                **cfg,
            )

            row = {"experiment": exp_name, "fold": fold}
            row.update(fold_metrics)
            fold_rows.append(row)

        exp_df = pd.DataFrame(fold_rows)

        summary = {
            "experiment": exp_name,
            "accuracy_mean": exp_df["accuracy"].mean(),
            "accuracy_std": exp_df["accuracy"].std(),
            "precision_mean": exp_df["precision"].mean(),
            "precision_std": exp_df["precision"].std(),
            "recall_mean": exp_df["recall"].mean(),
            "recall_std": exp_df["recall"].std(),
            "f1_mean": exp_df["f1"].mean(),
            "f1_std": exp_df["f1"].std(),
            "auc_mean": exp_df["auc"].mean(),
            "auc_std": exp_df["auc"].std(),
            "ap_mean": exp_df["ap"].mean(),
            "ap_std": exp_df["ap"].std(),
        }

        print("\nMean ± Std:")
        print(summary)
        all_results.append(summary)

    results_df = (
        pd.DataFrame(all_results)
        .sort_values("f1_mean", ascending=False)
        .reset_index(drop=True)
    )

    print("\n========== 消融总表 ==========")
    print(results_df)

    return results_df


def run_paper_aligned_cv_experiment(
    df_all,
    meta_cols,
    city_col,
    tokenizer,
    device,
    model_name="hfl/chinese-roberta-wwm-ext",
    n_splits=10,
    random_state=42,
):
    balanced_df = apply_nearmiss_by_city(
        df_all,
        feature_cols=meta_cols,
        city_col=city_col,
        label_col="label",
        random_state=random_state,
    )

    print("\n========== 平衡后数据分布 ==========")
    print(balanced_df["label"].value_counts())
    if city_col in balanced_df.columns:
        print("\n按 city 查看样本数（前几项）：")
        print(balanced_df[city_col].value_counts().head())

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_rows = []
    oof_probs = np.zeros(len(balanced_df), dtype=np.float32)
    oof_labels = balanced_df["label"].values.astype(int)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(balanced_df["text"].astype(str).values, balanced_df["label"].values),
        start=1,
    ):
        print(f"\n================ Fold {fold}/{n_splits} ================\n")

        train_df = balanced_df.iloc[train_idx].reset_index(drop=True)
        val_df = balanced_df.iloc[val_idx].reset_index(drop=True)

        fold_metrics, artifacts = run_single_fold(
            train_df=train_df,
            val_df=val_df,
            tokenizer=tokenizer,
            meta_cols=meta_cols,
            device=device,
            model_name=model_name,
            hidden_dim=256,
            num_claims=4,
            dropout=0.1,
            diversity_lambda=0.005,
            max_length=160,
            random_state=random_state + fold,
        )

        print("Fold metrics:", fold_metrics)

        oof_probs[val_idx] = artifacts["probs_pos"]

        row = {"fold": fold}
        row.update(fold_metrics)
        fold_rows.append(row)

    results_df = pd.DataFrame(fold_rows)

    print("\n========== 10-fold CV 结果 ==========")
    print(results_df)

    print("\n========== Mean ± Std ==========")
    for metric in ["accuracy", "precision", "recall", "f1", "auc", "ap"]:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

    overall_metrics = metrics_from_probs(oof_labels, oof_probs, threshold=0.5)
    print("\n========== OOF Overall ==========")
    print(overall_metrics)

    return balanced_df, results_df, oof_labels, oof_probs
