import numpy as np
import pandas as pd
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def apply_nearmiss_by_city(df, feature_cols, city_col="city", label_col="label", random_state=42):
    """
    论文对齐版：优先按 city 分组做 NearMiss-1。
    如果 city 只有一个值，就退化为全局 NearMiss-1。
    """
    df = df.reset_index(drop=True).copy()

    scaler = StandardScaler()
    x_full = scaler.fit_transform(df[feature_cols].astype(float).fillna(0).values)
    y_full = df[label_col].values

    if city_col not in df.columns or df[city_col].nunique() <= 1:
        print("⚠️ 没有有效的 city 列，退化为全局 NearMiss-1。")
        nm = NearMiss(version=1)
        _ = nm.fit_resample(x_full, y_full)
        selected_idx = nm.sample_indices_
        balanced_df = df.iloc[selected_idx].reset_index(drop=True)
        return balanced_df

    parts = []
    for _, sub_df in df.groupby(city_col):
        sub_df = sub_df.reset_index(drop=True)
        if sub_df[label_col].nunique() < 2:
            continue

        x_city = scaler.fit_transform(sub_df[feature_cols].astype(float).fillna(0).values)
        y_city = sub_df[label_col].values

        nm = NearMiss(version=1)
        _ = nm.fit_resample(x_city, y_city)
        selected_idx = nm.sample_indices_

        parts.append(sub_df.iloc[selected_idx])

    balanced_df = pd.concat(parts, axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return balanced_df


def fit_meta_scaler(train_df, val_df, meta_cols):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[meta_cols].astype(float).fillna(0).values)
    x_val = scaler.transform(val_df[meta_cols].astype(float).fillna(0).values)
    return x_train, x_val, scaler


def build_rf_oof_scores(x_train, y_train, x_val, random_state=42):
    """
    对训练集做 OOF RF 概率，防止泄漏；
    对验证集用 full-train RF 打分。
    """
    inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_scores = np.zeros(len(x_train), dtype=np.float32)

    for inner_tr_idx, inner_val_idx in inner_skf.split(x_train, y_train):
        x_tr, x_oof = x_train[inner_tr_idx], x_train[inner_val_idx]
        y_tr = y_train[inner_tr_idx]

        rf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(x_tr, y_tr)
        train_scores[inner_val_idx] = rf.predict_proba(x_oof)[:, 1]

    rf_full = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    rf_full.fit(x_train, y_train)
    val_scores = rf_full.predict_proba(x_val)[:, 1].astype(np.float32)

    return train_scores, val_scores, rf_full
