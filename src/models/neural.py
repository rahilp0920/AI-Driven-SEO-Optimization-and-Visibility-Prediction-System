"""PyTorch MLP classifier — neural family (model #4). Trained in Colab,
checkpoint exported via the rubric §VI snippet pattern (state_dict + config),
loaded locally for evaluation and dashboard inference.

The checkpoint payload is::

    {
        "model_state_dict": <state dict>,
        "config": {"input_dim": ..., "hidden_dims": [...], "dropout": ...},
        "scaler_mean": <numpy array>,
        "scaler_scale": <numpy array>,
        "feature_names": [...],          # column order at training time
        "epoch": <int>,
        "best_threshold": <float>,
    }

`MLPInferenceWrapper` exposes a sklearn-shaped ``predict`` / ``predict_proba``
so the comparison notebook, SHAP explainer, and dashboard can treat the MLP
identically to LR / RF / XGB.

CLI:
    python -m src.models.neural        # train locally + write checkpoint
    python -m src.models.neural --evaluate-only --checkpoint models/mlp_checkpoint.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.models.evaluate import (
    evaluate_classifier,
    load_features,
    save_metrics,
    stratified_split,
)

LOG = logging.getLogger("mlp")


class MLP(nn.Module):
    """Feed-forward classifier head: input → [hidden_dims with ReLU + Dropout] → 1 logit."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (128, 64), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPInferenceWrapper:
    """sklearn-shaped wrapper so the rest of the pipeline (SHAP, dashboard,
    comparison notebook) can call ``predict`` / ``predict_proba`` without
    knowing it's a torch model."""

    def __init__(
        self,
        model: MLP,
        scaler: StandardScaler,
        feature_names: list[str],
        threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.scaler = scaler
        self.feature_names = feature_names
        self.threshold = threshold
        self.device = device

    def _prepare(self, X: pd.DataFrame | np.ndarray) -> torch.Tensor:
        if isinstance(X, pd.DataFrame):
            X = X.reindex(columns=self.feature_names, fill_value=0.0).to_numpy()
        Xs = self.scaler.transform(X)
        return torch.from_numpy(Xs.astype(np.float32)).to(self.device)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = self.model(self._prepare(X))
        p1 = torch.sigmoid(logits).cpu().numpy()
        return np.stack([1.0 - p1, p1], axis=1)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def _train_loop(
    model: MLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    pos_weight: float,
    device: str,
) -> tuple[MLP, float]:
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_w = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    X_train_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(device)

    n = len(X_train_t)
    best_val = float("inf")
    best_state: dict | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        train_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            optim.zero_grad()
            logits = model(X_train_t[idx])
            loss = loss_fn(logits, y_train_t[idx])
            loss.backward()
            optim.step()
            train_loss += float(loss) * len(idx)
        train_loss /= n

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val_t), y_val_t))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            LOG.info("epoch %d/%d  train_loss=%.4f  val_loss=%.4f", epoch, epochs, train_loss, val_loss)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def train(
    csv_path: Path = Path("data/processed/features.csv"),
    out_checkpoint: Path = Path("models/mlp_checkpoint.pt"),
    out_metrics: Path = Path("models/metrics/mlp.json"),
    hidden_dims: tuple[int, ...] = (128, 64),
    dropout: float = 0.2,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    random_state: int = 42,
    device: str | None = None,
) -> dict:
    """Train an MLP on the local feature matrix. The same function runs in
    Colab — point ``csv_path`` at a Drive-mounted copy of features.csv and
    the resulting checkpoint loads identically on the local machine."""
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG.info("device=%s", device)

    X, y, _ = load_features(csv_path)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = stratified_split(X, y, random_state=random_state)
    X_inner, X_val, y_inner, y_val = stratified_split(
        X_train, y_train, test_size=0.15, random_state=random_state
    )

    scaler = StandardScaler()
    X_inner_s = scaler.fit_transform(X_inner)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    pos = float(y_inner.sum())
    neg = float(len(y_inner) - pos)
    pos_weight = (neg / pos) if pos > 0 else 1.0

    model = MLP(input_dim=X_inner_s.shape[1], hidden_dims=hidden_dims, dropout=dropout)
    model, best_val_loss = _train_loop(
        model, X_inner_s, y_inner.to_numpy(), X_val_s, y_val.to_numpy(),
        epochs=epochs, batch_size=batch_size, lr=lr, pos_weight=pos_weight, device=device,
    )
    LOG.info("best val loss=%.4f", best_val_loss)

    wrapper = MLPInferenceWrapper(model, scaler, feature_names, threshold=0.5, device=device)
    y_pred = wrapper.predict(X_test_s)
    y_proba = wrapper.predict_proba(X_test_s)[:, 1]

    metrics = evaluate_classifier("mlp", y_test, y_pred, y_proba)
    save_metrics(metrics, out_metrics)

    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"input_dim": X_inner_s.shape[1], "hidden_dims": list(hidden_dims), "dropout": dropout},
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "feature_names": feature_names,
            "epoch": epochs,
            "best_threshold": 0.5,
        },
        out_checkpoint,
    )
    LOG.info("checkpoint → %s", out_checkpoint)

    return {"metrics": metrics, "best_val_loss": best_val_loss, "checkpoint": str(out_checkpoint)}


def load_checkpoint(path: Path, device: str = "cpu") -> MLPInferenceWrapper:
    """Load a checkpoint saved by ``train`` (or by the Colab equivalent) and
    reconstruct a sklearn-shaped inference wrapper. This is the function the
    dashboard calls at startup."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = MLP(input_dim=cfg["input_dim"], hidden_dims=tuple(cfg["hidden_dims"]), dropout=cfg["dropout"])
    model.load_state_dict(ckpt["model_state_dict"])

    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    return MLPInferenceWrapper(
        model=model,
        scaler=scaler,
        feature_names=ckpt["feature_names"],
        threshold=ckpt.get("best_threshold", 0.5),
        device=device,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train (or evaluate) the MLP classifier.")
    p.add_argument("--csv", type=Path, default=Path("data/processed/features.csv"))
    p.add_argument("--checkpoint", type=Path, default=Path("models/mlp_checkpoint.pt"))
    p.add_argument("--out-metrics", type=Path, default=Path("models/metrics/mlp.json"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--evaluate-only", action="store_true", help="Skip training; load --checkpoint and score test split.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.evaluate_only:
        wrapper = load_checkpoint(args.checkpoint)
        X, y, _ = load_features(args.csv)
        _, X_test, _, y_test = stratified_split(X, y)
        y_pred = wrapper.predict(X_test)
        y_proba = wrapper.predict_proba(X_test)[:, 1]
        metrics = evaluate_classifier("mlp", y_test, y_pred, y_proba)
        save_metrics(metrics, args.out_metrics)
        print(f"mlp (eval-only): f1={metrics.f1:.4f} roc_auc={metrics.roc_auc:.4f}")
        return 0

    out = train(
        csv_path=args.csv,
        out_checkpoint=args.checkpoint,
        out_metrics=args.out_metrics,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(f"mlp: f1={out['metrics'].f1:.4f} roc_auc={out['metrics'].roc_auc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
