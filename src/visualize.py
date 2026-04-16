import torch

from .data_prep import split_into_clauses


def visualize_clause_claims(text, model, tokenizer, device, max_length=128):
    model.eval()

    clauses = split_into_clauses(text)
    clause_mask = torch.tensor([[1] * len(clauses)], dtype=torch.long).to(device)

    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    token_type_ids = encoded.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            clauses=[clauses],
            clause_mask=clause_mask,
            return_extras=True,
        )

    claim_attn = outputs["claim_attn"][0]
    claim_weights = outputs["claim_weights"][0]
    gate = outputs["gate"][0].item()
    logits = outputs["output"].logits[0]
    pred = torch.argmax(logits).item()

    print("=" * 90)
    print("原始文本：")
    print(text)
    print("-" * 90)
    print(f"分句结果: {clauses}")
    print(f"预测类别: {pred}")
    print(f"gate: {gate:.4f}")
    print("=" * 90)

    for i in range(claim_attn.size(0)):
        weights = claim_attn[i][: len(clauses)].detach().cpu()

        ranked = sorted(
            [(clauses[j], weights[j].item()) for j in range(len(clauses))],
            key=lambda x: x[1],
            reverse=True,
        )

        print(f"\n[Claim {i + 1}] claim_weight={claim_weights[i].item():.4f}")
        for rank, (clause, val) in enumerate(ranked, 1):
            print(f"{rank:02d}. clause={clause:<20} weight={val:.4f}")
