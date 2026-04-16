import numpy as np
import torch
from torch.utils.data import Dataset


class ClaimMetaDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        meta_matrix,
        rf_scores,
        max_length=160
    ):
        self.df = df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.max_length = max_length

        texts = self.df["text"].astype(str).tolist()
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.token_type_ids = enc.get("token_type_ids", None)

        self.labels = self.df["label"].astype(int).tolist()
        self.clauses = self.df["clauses"].tolist()

        self.meta_matrix = meta_matrix.astype(np.float32)
        self.rf_scores = rf_scores.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
            "clauses": self.clauses[idx],
            "meta": self.meta_matrix[idx],
            "rf_score": self.rf_scores[idx]
        }

        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]

        return item


def claim_meta_collator(features):
    batch = {}

    batch["input_ids"] = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    batch["attention_mask"] = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)

    if "token_type_ids" in features[0]:
        batch["token_type_ids"] = torch.tensor([f["token_type_ids"] for f in features], dtype=torch.long)

    batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long)

    clauses_list = [f["clauses"] for f in features]
    max_clauses = max(len(x) for x in clauses_list)

    padded_clauses = []
    clause_mask = []

    for clauses in clauses_list:
        pad_len = max_clauses - len(clauses)
        padded_clauses.append(clauses + [""] * pad_len)
        clause_mask.append([1] * len(clauses) + [0] * pad_len)

    batch["clauses"] = padded_clauses
    batch["clause_mask"] = torch.tensor(clause_mask, dtype=torch.long)

    batch["meta"] = torch.tensor([f["meta"] for f in features], dtype=torch.float)
    batch["rf_score"] = torch.tensor([f["rf_score"] for f in features], dtype=torch.float).unsqueeze(-1)

    return batch
