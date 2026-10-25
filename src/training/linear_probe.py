"""
SigLIP and CLIP Linear Probing and Lightweight Fine-Tuning Engine.
Trains calibrated linear classification heads atop frozen foundation visual embeddings
with temperature scaling and Expected Calibration Error (ECE) monitoring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field

logger = logging.getLogger("OmniVision.Training.LinearProbe")


class ProbeEpochMetrics(BaseModel):
    """Evaluation metrics per training epoch."""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    expected_calibration_error: Optional[float] = None


class LinearProbeHead(nn.Module):
    """Linear classification projection head with learnable temperature scaling."""

    def __init__(self, embed_dim: int = 768, num_classes: int = 10, init_temp: float = 1.0) -> None:
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes)
        # Learnable log temperature for calibrated probability output
        self.log_temp = nn.Parameter(torch.tensor(float(np.log(init_temp))))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute temperature-scaled logits."""
        # Normalized embeddings
        z = F.normalize(embeddings, p=2, dim=-1)
        logits = self.linear(z)
        temp = torch.exp(self.log_temp).clamp(min=0.01, max=10.0)
        return logits / temp


def compute_ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE):
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(targets)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(targets)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


class SigLIPLinearProbeTrainer:
    """
    Linear Probe Domain Adapter.
    Trains and calibrates classification heads atop frozen SigLIP/CLIP visual embeddings.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 10,
        class_names: Optional[List[str]] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

        self.model = LinearProbeHead(embed_dim=embed_dim, num_classes=num_classes)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self._history: List[ProbeEpochMetrics] = []

    def fit(
        self,
        train_embeddings: torch.Tensor,
        train_labels: torch.Tensor,
        val_embeddings: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        num_epochs: int = 10,
        batch_size: int = 32,
    ) -> List[ProbeEpochMetrics]:
        """
        Train the linear probe head over extracted backbone visual embeddings.
        """
        n_samples = train_embeddings.shape[0]
        dataset = torch.utils.data.TensorDataset(train_embeddings, train_labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0

            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = F.cross_entropy(logits, batch_y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * batch_x.size(0)
                preds = logits.argmax(dim=-1)
                correct += preds.eq(batch_y).sum().item()

            epoch_loss = running_loss / max(1, n_samples)
            epoch_acc = correct / max(1, n_samples)

            # Validation evaluation if validation data provided
            val_loss, val_acc, ece = None, None, None
            if val_embeddings is not None and val_labels is not None:
                val_eval = self.evaluate(val_embeddings, val_labels)
                val_loss = val_eval["loss"]
                val_acc = val_eval["accuracy"]
                ece = val_eval["ece"]

            metrics = ProbeEpochMetrics(
                epoch=epoch + 1,
                train_loss=round(epoch_loss, 4),
                train_accuracy=round(epoch_acc, 4),
                val_loss=round(val_loss, 4) if val_loss is not None else None,
                val_accuracy=round(val_acc, 4) if val_acc is not None else None,
                expected_calibration_error=round(ece, 4) if ece is not None else None,
            )
            self._history.append(metrics)

        logger.info(f"Finished linear probe training ({num_epochs} epochs). Final train acc: {self._history[-1].train_accuracy*100:.1f}%")
        return self._history

    def predict(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class indices and calibrated softmax probabilities.
        Returns: (predicted_classes [N], probabilities [N, num_classes])
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(embeddings)
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
        return preds, probs

    def evaluate(self, embeddings: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute evaluation metrics (accuracy, loss, ECE)."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(embeddings)
            loss = F.cross_entropy(logits, targets).item()
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            acc = preds.eq(targets).float().mean().item()
            ece = compute_ece(probs, targets)

        return {
            "loss": float(loss),
            "accuracy": float(acc),
            "ece": float(ece),
        }
