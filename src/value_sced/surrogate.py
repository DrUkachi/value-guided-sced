"""Compact, reproducible MLP ensemble training and prediction."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class DispatchMLP(nn.Module):
    """Fully connected dispatch regressor."""

    def __init__(self, input_size: int, output_size: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        sizes = (input_size,) + hidden_sizes + (output_size,)
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            layers.append(nn.Linear(left, right))
            if index < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


@dataclass
class TrainedEnsemble:
    """Models plus source-fitted transformations and training evidence."""

    models: tuple[DispatchMLP, ...]
    input_mean: np.ndarray
    input_scale: np.ndarray
    output_mean: np.ndarray
    output_scale: np.ndarray
    best_epochs: tuple[int, ...]
    validation_losses: tuple[float, ...]

    def predict_members(self, features: np.ndarray) -> np.ndarray:
        values = (np.asarray(features, dtype=np.float32) - self.input_mean) / self.input_scale
        tensor = torch.as_tensor(values, dtype=torch.float32)
        predictions = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                scaled = model(tensor).cpu().numpy()
                predictions.append(scaled * self.output_scale + self.output_mean)
        return np.stack(predictions)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_members(features).mean(axis=0)


def _safe_scale(values: np.ndarray) -> np.ndarray:
    scale = values.std(axis=0)
    return np.where(scale > 1e-12, scale, 1.0)


def source_train_validation_indices(
    number_samples: int, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible source-only split."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    order = rng.permutation(number_samples)
    validation_size = max(1, int(round(number_samples * validation_fraction)))
    return order[validation_size:], order[:validation_size]


def train_ensemble(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    ensemble_size: int,
    hidden_sizes: tuple[int, ...],
    learning_rate: float,
    max_epochs: int,
    patience: int,
    seed: int,
    scaling_features: np.ndarray | None = None,
    scaling_targets: np.ndarray | None = None,
) -> TrainedEnsemble:
    """Train an ensemble with source-fitted scaling and early stopping."""

    x_train = np.asarray(train_features, dtype=np.float32)
    y_train = np.asarray(train_targets, dtype=np.float32)
    x_validation = np.asarray(validation_features, dtype=np.float32)
    y_validation = np.asarray(validation_targets, dtype=np.float32)
    scale_x = x_train if scaling_features is None else np.asarray(scaling_features, dtype=np.float32)
    scale_y = y_train if scaling_targets is None else np.asarray(scaling_targets, dtype=np.float32)
    input_mean, output_mean = scale_x.mean(axis=0), scale_y.mean(axis=0)
    input_scale, output_scale = _safe_scale(scale_x), _safe_scale(scale_y)
    x_train_tensor = torch.as_tensor((x_train - input_mean) / input_scale)
    y_train_tensor = torch.as_tensor((y_train - output_mean) / output_scale)
    x_validation_tensor = torch.as_tensor((x_validation - input_mean) / input_scale)
    y_validation_tensor = torch.as_tensor((y_validation - output_mean) / output_scale)

    models, best_epochs, validation_losses = [], [], []
    for member in range(ensemble_size):
        member_seed = seed + member
        torch.manual_seed(member_seed)
        model = DispatchMLP(x_train.shape[1], y_train.shape[1], hidden_sizes)
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = nn.MSELoss()
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        best_epoch = 0
        stale_epochs = 0
        for epoch in range(max_epochs):
            model.train()
            optimiser.zero_grad()
            loss = loss_function(model(x_train_tensor), y_train_tensor)
            loss.backward()
            optimiser.step()
            model.eval()
            with torch.no_grad():
                validation_loss = float(
                    loss_function(model(x_validation_tensor), y_validation_tensor)
                )
            if validation_loss < best_loss - 1e-8:
                best_loss = validation_loss
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break
        model.load_state_dict(best_state)
        models.append(model)
        best_epochs.append(best_epoch)
        validation_losses.append(best_loss)

    return TrainedEnsemble(
        tuple(models), input_mean, input_scale, output_mean, output_scale,
        tuple(best_epochs), tuple(validation_losses)
    )
