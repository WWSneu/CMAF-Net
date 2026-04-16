import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


class ClauseClaimExtractor(nn.Module):
    def __init__(self, hidden_dim=256, num_claims=4, temperature=0.7):
        super().__init__()
        self.num_claims = num_claims
        self.temperature = temperature

        self.claim_queries = nn.Parameter(torch.randn(num_claims, hidden_dim))
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, clause_states, clause_mask=None):
        """
        clause_states: [B, C, H]
        clause_mask:   [B, C]  1表示有效子句，0表示padding
        """
        bsz, num_clauses, dim = clause_states.shape

        q = self.claim_queries.unsqueeze(0).expand(bsz, -1, -1)
        k = self.key_proj(clause_states)
        v = self.value_proj(clause_states)

        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(dim)
        scores = scores / self.temperature

        if clause_mask is not None:
            scores = scores.masked_fill(~clause_mask.bool().unsqueeze(1), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        claims = torch.matmul(attn, v)
        claims = self.norm(claims)

        return claims, attn, scores


def compute_claim_diversity_loss(claim_attn, valid_mask):
    """
    claim_attn: [B, K, C]
    K = claim数
    C = clause数
    希望不同 claim 的注意力不要重合太多
    """
    mask = valid_mask.unsqueeze(1).float()
    attn = claim_attn * mask
    attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

    overlap = torch.matmul(attn, attn.transpose(1, 2))

    k = overlap.size(1)
    eye = torch.eye(k, device=overlap.device).unsqueeze(0)
    off_diag = overlap * (1 - eye)

    return off_diag.mean()


def compute_query_orth_loss(claim_queries):
    """
    claim_queries: [K, H]
    希望 query 本身方向不要太像
    """
    q = F.normalize(claim_queries, p=2, dim=-1)
    sim = torch.matmul(q, q.transpose(0, 1))
    k = sim.size(0)
    eye = torch.eye(k, device=sim.device)
    off_diag = sim * (1 - eye)
    return off_diag.pow(2).mean()


def encode_clauses_batch(clauses_batch, tokenizer, encoder, device, max_length=32):
    """
    clauses_batch: List[List[str]]
    返回:
        clause_embeddings: [B, C, H_roberta]
    """
    bsz = len(clauses_batch)
    num_clauses = len(clauses_batch[0])

    flat_clauses = []
    for clauses in clauses_batch:
        for clause in clauses:
            flat_clauses.append(clause if clause.strip() else "[PAD_CLAUSE]")

    enc = tokenizer(
        flat_clauses,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    outputs = encoder(**enc)
    cls_emb = outputs.last_hidden_state[:, 0, :]

    clause_embeddings = cls_emb.view(bsz, num_clauses, -1)
    return clause_embeddings


class RobertaClaimMetaFusionAblation(nn.Module):
    def __init__(
        self,
        model_name="hfl/chinese-roberta-wwm-ext",
        tokenizer=None,
        meta_dim=24,
        hidden_dim=256,
        num_claims=4,
        num_labels=2,
        dropout=0.1,
        temperature=0.7,
        diversity_lambda=0.005,
        use_cls=True,
        use_claim=True,
        use_meta=True,
        use_rf=True
    ):
        super().__init__()
        if not (use_cls or use_claim or use_meta or use_rf):
            raise ValueError("至少要保留一个分支。")

        self.tokenizer = tokenizer
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=False)
        roberta_hidden = self.encoder.config.hidden_size

        self.hidden_dim = hidden_dim
        self.meta_dim = meta_dim
        self.diversity_lambda = diversity_lambda

        self.use_cls = use_cls
        self.use_claim = use_claim
        self.use_meta = use_meta
        self.use_rf = use_rf

        self.token_proj = nn.Linear(roberta_hidden, hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)

        self.cls_proj = nn.Linear(roberta_hidden, hidden_dim)

        self.claim_extractor = ClauseClaimExtractor(
            hidden_dim=hidden_dim,
            num_claims=num_claims,
            temperature=temperature
        )

        self.claim_importance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.meta_attention = nn.Sequential(
            nn.Linear(meta_dim, meta_dim),
            nn.Tanh(),
            nn.Linear(meta_dim, meta_dim),
            nn.Sigmoid()
        )

        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        fusion_dim = 0
        if self.use_cls:
            fusion_dim += hidden_dim
        if self.use_claim:
            fusion_dim += hidden_dim
        if self.use_meta:
            fusion_dim += hidden_dim
        if self.use_rf:
            fusion_dim += 1

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        labels=None,
        clauses=None,
        clause_mask=None,
        meta=None,
        rf_score=None,
        return_extras=False,
        **kwargs
    ):
        if self.use_meta and meta is None:
            raise ValueError("meta 不能为空。")
        if self.use_claim and clauses is None:
            raise ValueError("clauses 不能为空。")
        if self.use_claim and clause_mask is None:
            raise ValueError("开启 claim 分支时，clause_mask 不能为空。")
        if self.use_rf and rf_score is None:
            rf_score = torch.zeros(input_ids.size(0), 1, device=input_ids.device)

        cls_feat = None
        if self.use_cls:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            cls_state = outputs.last_hidden_state[:, 0, :]
            cls_feat = self.cls_proj(cls_state)

        claim_attn = None
        claim_weights = None
        raw_scores = None
        gate = None
        claim_used = None
        if self.use_claim:
            clause_emb_raw = encode_clauses_batch(
                clauses_batch=clauses,
                tokenizer=self.tokenizer,
                encoder=self.encoder,
                device=input_ids.device,
                max_length=32
            )
            clause_states = self.token_proj(clause_emb_raw)
            clause_states = self.token_norm(clause_states)

            claim_states, claim_attn, raw_scores = self.claim_extractor(
                clause_states=clause_states,
                clause_mask=clause_mask
            )

            claim_logits = self.claim_importance(claim_states).squeeze(-1)
            claim_weights = F.softmax(claim_logits, dim=-1)

            fused_claim = torch.sum(
                claim_states * claim_weights.unsqueeze(-1),
                dim=1
            )

            if self.use_cls:
                gate_input = torch.cat([cls_feat, fused_claim], dim=-1)
                gate = self.gate(gate_input)
            else:
                gate = torch.ones(fused_claim.size(0), 1, device=fused_claim.device)

            claim_used = fused_claim * (0.25 + 0.50 * gate)

        meta_attn_weights = None
        meta_feat = None
        if self.use_meta:
            meta_attn_weights = self.meta_attention(meta)
            meta_weighted = meta * meta_attn_weights
            meta_feat = self.meta_encoder(meta_weighted)

        parts = []
        if self.use_cls:
            parts.append(cls_feat)
        if self.use_claim:
            parts.append(claim_used)
        if self.use_meta:
            parts.append(meta_feat)
        if self.use_rf:
            parts.append(rf_score)

        final_feat = torch.cat(parts, dim=-1)

        logits = self.classifier(final_feat)

        loss = None
        if labels is not None:
            cls_loss = F.cross_entropy(logits, labels)
            if self.use_claim:
                diversity_loss = compute_claim_diversity_loss(claim_attn, clause_mask)
                loss = cls_loss + self.diversity_lambda * diversity_loss
            else:
                loss = cls_loss

        output = SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )

        if return_extras:
            return {
                "output": output,
                "claim_attn": claim_attn,
                "claim_weights": claim_weights,
                "gate": gate,
                "raw_scores": raw_scores,
                "clauses": clauses,
                "meta_attn_weights": meta_attn_weights,
                "rf_score": rf_score
            }

        return output


class RobertaClaimMetaFusion(RobertaClaimMetaFusionAblation):
    def __init__(
        self,
        model_name="hfl/chinese-roberta-wwm-ext",
        tokenizer=None,
        meta_dim=24,
        hidden_dim=256,
        num_claims=4,
        num_labels=2,
        dropout=0.1,
        temperature=0.7,
        diversity_lambda=0.005
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=tokenizer,
            meta_dim=meta_dim,
            hidden_dim=hidden_dim,
            num_claims=num_claims,
            num_labels=num_labels,
            dropout=dropout,
            temperature=temperature,
            diversity_lambda=diversity_lambda,
            use_cls=True,
            use_claim=True,
            use_meta=True,
            use_rf=True
        )
