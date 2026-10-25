"""
Student Detector Knowledge Distillation Engine.
Distills complex ensembles (YOLO11 + RT-DETR + WBF) into a lightweight,
low-latency student detector using soft KL-divergence, hard CE, and smooth L1 bounding box regression.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field

logger = logging.getLogger("OmniVision.Training.Distillation")


class DistillationLossOutput(BaseModel):
    """Component breakdown of distillation loss."""
    total_loss: float
    hard_label_loss: float
    soft_teacher_loss: float
    box_regression_loss: float


class StudentDetectorNetwork(nn.Module):
    """Lightweight student object detection backbone and prediction heads."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10, base_dim: int = 32) -> None:
        super().__init__()
        # Feature extraction backbone (fast depthwise-separable conv layers)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.ReLU6(),
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_dim * 2),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # Classification Head
        self.cls_head = nn.Sequential(
            nn.Linear(base_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        # Bounding Box Regression Head [x1, y1, x2, y2]
        self.bbox_head = nn.Sequential(
            nn.Linear(base_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid(),  # normalized [0, 1] bounds
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns: (cls_logits [B, num_classes], bbox_coords [B, 4])
        """
        feats = self.backbone(x)
        logits = self.cls_head(feats)
        boxes = self.bbox_head(feats)
        return logits, boxes


class StudentDetectorDistiller:
    """
    Teacher-to-Student Distillation Orchestrator.
    Minimizes soft label divergence against the teacher ensemble and aligns spatial coordinates.
    """

    def __init__(
        self,
        num_classes: int = 10,
        temperature: float = 3.0,
        alpha_hard: float = 0.3,
        beta_bbox: float = 1.5,
        learning_rate: float = 5e-4,
    ) -> None:
        self.temperature = temperature
        self.alpha_hard = alpha_hard
        self.beta_bbox = beta_bbox

        self.student = StudentDetectorNetwork(num_classes=num_classes)
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=learning_rate)

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        student_boxes: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_boxes: torch.Tensor,
        true_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, DistillationLossOutput]:
        """
        L = alpha * L_hard + (1 - alpha) * T^2 * KL(student/T, teacher/T) + beta * SmoothL1(b_student, b_teacher)
        """
        t = self.temperature

        # 1. Soft Teacher Distillation Loss (KL divergence over softened logits)
        p_student = F.log_softmax(student_logits / t, dim=-1)
        p_teacher = F.softmax(teacher_logits / t, dim=-1)
        loss_soft = F.kl_div(p_student, p_teacher, reduction="batchmean") * (t ** 2)

        # 2. Hard Label Cross-Entropy Loss (if ground-truth labels provided)
        if true_labels is not None:
            loss_hard = F.cross_entropy(student_logits, true_labels)
        else:
            loss_hard = torch.tensor(0.0, device=student_logits.device)

        # 3. Bounding Box Coordinate Regression Loss (Smooth L1)
        loss_box = F.smooth_l1_loss(student_boxes, teacher_boxes)

        # Aggregate total distillation loss
        total_loss = (self.alpha_hard * loss_hard) + ((1.0 - self.alpha_hard) * loss_soft) + (self.beta_bbox * loss_box)

        breakdown = DistillationLossOutput(
            total_loss=round(float(total_loss.item()), 4),
            hard_label_loss=round(float(loss_hard.item()), 4),
            soft_teacher_loss=round(float(loss_soft.item()), 4),
            box_regression_loss=round(float(loss_box.item()), 4),
        )
        return total_loss, breakdown

    def train_step(
        self,
        images: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_boxes: torch.Tensor,
        true_labels: Optional[torch.Tensor] = None,
    ) -> DistillationLossOutput:
        """
        Execute forward pass, backpropagation, and weight update for student network.
        """
        self.student.train()
        self.optimizer.zero_grad()

        student_logits, student_boxes = self.student(images)
        loss, breakdown = self.compute_loss(
            student_logits=student_logits,
            student_boxes=student_boxes,
            teacher_logits=teacher_logits,
            teacher_boxes=teacher_boxes,
            true_labels=true_labels,
        )

        loss.backward()
        self.optimizer.step()
        return breakdown
