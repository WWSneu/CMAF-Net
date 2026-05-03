import argparse
import os

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from transformers import AutoTokenizer

from src.data_prep import build_feature_dataframe
from src.experiment import (
    run_ablation_cv_experiment,
    run_paper_aligned_cv_experiment,
    run_strict_cv_experiment,
)
from src.meta_sampling import apply_nearmiss_by_city
from src.training_utils import search_best_threshold, set_global_seed


def configure_env(cuda_visible_devices="6"):
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 检测到 CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 检测到 MPS")
    else:
        device = torch.device("cpu")
        print("⚠️ 使用 CPU")
    return device


def plot_oof_curves(oof_labels, oof_probs, output_dir, file_prefix="oof"):
    fpr, tpr, _ = roc_curve(oof_labels, oof_probs)
    precision_curve, recall_curve, _ = precision_recall_curve(oof_labels, oof_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(oof_labels, oof_probs):.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (OOF)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_roc.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(
        recall_curve,
        precision_curve,
        label=f"PR-AUC = {average_precision_score(oof_labels, oof_probs):.4f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (OOF)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_pr.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data/merged_data.csv")
    parser.add_argument("--model_name", type=str, default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--cuda_visible_devices", type=str, default="6")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_ablation", action="store_true", help="在主实验后运行系统消融")
    parser.add_argument("--ablation_quick", action="store_true", help="只跑 4 组核心消融")
    parser.add_argument("--ablation_only", action="store_true", help="只跑消融，不跑主实验")
    parser.add_argument("--ablation_n_splits", type=int, default=None)
    parser.add_argument(
        "--strict_cv",
        action="store_true",
        help="每个fold内部只对训练集做NearMiss，验证集保持原始分布",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    configure_env(cuda_visible_devices=args.cuda_visible_devices)
    set_global_seed(args.random_state)

    df_all, meta_cols, city_col = build_feature_dataframe(args.csv_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device = get_device()

    balanced_df = None
    if not args.ablation_only:
        if args.strict_cv:
            strict_results_df, strict_oof_labels, strict_oof_probs = run_strict_cv_experiment(
                df_all=df_all,
                meta_cols=meta_cols,
                city_col=city_col,
                tokenizer=tokenizer,
                device=device,
                model_name=args.model_name,
                n_splits=args.n_splits,
                random_state=args.random_state,
            )

            best_t, best_metrics = search_best_threshold(strict_oof_labels, strict_oof_probs)
            print("Strict CV best threshold:", round(best_t, 4))
            print(best_metrics)

            strict_results_df.to_csv(
                os.path.join(args.output_dir, "strict_cv_results.csv"),
                index=False,
            )
            plot_oof_curves(
                strict_oof_labels,
                strict_oof_probs,
                args.output_dir,
                file_prefix="strict_oof",
            )
        else:
            balanced_df, cv_results_df, oof_labels, oof_probs = run_paper_aligned_cv_experiment(
                df_all=df_all,
                meta_cols=meta_cols,
                city_col=city_col,
                tokenizer=tokenizer,
                device=device,
                model_name=args.model_name,
                n_splits=args.n_splits,
                random_state=args.random_state,
            )

            best_t, best_metrics = search_best_threshold(oof_labels, oof_probs)
            print("Best threshold:", round(best_t, 4))
            print(best_metrics)

            cv_results_df.to_csv(os.path.join(args.output_dir, "cv_results.csv"), index=False)
            balanced_df.to_csv(os.path.join(args.output_dir, "balanced_data.csv"), index=False)
            plot_oof_curves(oof_labels, oof_probs, args.output_dir, file_prefix="oof")

    if args.run_ablation or args.ablation_only:
        if args.strict_cv:
            print("⚠️ 当前消融实验仍按 paper-aligned 协议运行（先全局 NearMiss，再CV）。")
        if balanced_df is None:
            balanced_df = apply_nearmiss_by_city(
                df_all,
                feature_cols=meta_cols,
                city_col=city_col,
                label_col="label",
                random_state=args.random_state,
            )

        ablation_n_splits = args.ablation_n_splits or args.n_splits
        ablation_results_df = run_ablation_cv_experiment(
            balanced_df=balanced_df,
            meta_cols=meta_cols,
            tokenizer=tokenizer,
            device=device,
            model_name=args.model_name,
            random_state=args.random_state,
            n_splits=ablation_n_splits,
            quick_mode=args.ablation_quick,
        )
        ablation_results_df.to_csv(
            os.path.join(args.output_dir, "ablation_results.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        print("消融结果已保存到:", os.path.join(args.output_dir, "ablation_results.csv"))

    print("训练完成，结果已保存到:", args.output_dir)


if __name__ == "__main__":
    main()
