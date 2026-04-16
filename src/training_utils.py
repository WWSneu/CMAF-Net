import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import TrainerCallback


def metrics_from_probs(labels, probs_pos, threshold=0.5):
    preds = (probs_pos >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }

    try:
        metrics["auc"] = roc_auc_score(labels, probs_pos)
    except Exception:
        metrics["auc"] = float("nan")

    try:
        metrics["ap"] = average_precision_score(labels, probs_pos)
    except Exception:
        metrics["ap"] = float("nan")

    return metrics


def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    if isinstance(logits, tuple):
        logits = logits[0]

    probs = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()
    probs_pos = probs[:, 1]

    return metrics_from_probs(labels, probs_pos, threshold=0.5)


class InMemoryEarlyStoppingCallback(TrainerCallback):
    def __init__(self, metric_name="eval_f1", patience=2, greater_is_better=True):
        self.metric_name = metric_name
        self.patience = patience
        self.greater_is_better = greater_is_better
        self.best_metric = None
        self.bad_epochs = 0
        self.best_state_dict = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.best_metric = None
        self.bad_epochs = 0
        self.best_state_dict = None
        return control

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if metrics is None or self.metric_name not in metrics:
            return control

        current = metrics[self.metric_name]

        if self.best_metric is None:
            improved = True
        else:
            if self.greater_is_better:
                improved = current > self.best_metric + 1e-6
            else:
                improved = current < self.best_metric - 1e-6

        if improved:
            self.best_metric = current
            self.bad_epochs = 0
            self.best_state_dict = {
                k: v.detach().cpu().clone().contiguous()
                for k, v in model.state_dict().items()
            }
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                control.should_training_stop = True

        return control

    def restore_best_weights(self, model):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)


def build_optimizer(model, encoder_lr=2e-5, head_lr=1e-4, weight_decay=0.01):
    encoder_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    return optimizer


def search_best_threshold(labels, probs):
    best_t = 0.5
    best_f1 = -1
    best_metrics = None

    for t in [i / 100 for i in range(5, 96)]:
        metrics = metrics_from_probs(labels, probs, threshold=t)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_t = t
            best_metrics = metrics

    return best_t, best_metrics
