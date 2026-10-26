"""普通 RNN / LSTM 五类心拍分类器。"""
from __future__ import annotations

import torch
from torch import nn


class ECGClassifier(nn.Module):
    """默认方案 B：两层 128 单元 LSTM，分类器 128→128→64→5。"""
    def __init__(self, model_type: str = "lstm", hidden_size: int = 128,
                 num_layers: int = 2, recurrent_dropout: float = 0.2,
                 classifier_dropout: float = 0.5, num_classes: int = 5,
                 pooling: str = "last", input_size: int = 300,
                 classifier_hidden_sizes: tuple[int, ...] | list[int] = (128, 64)):
        super().__init__()
        if model_type not in {"rnn", "lstm"}:
            raise ValueError("model_type must be rnn or lstm")
        if hidden_size < 1 or num_layers < 1:
            raise ValueError("hidden_size and num_layers must be positive")
        if not (0 <= recurrent_dropout < 1 and 0 <= classifier_dropout < 1):
            raise ValueError("Dropout probabilities must be in [0, 1)")
        if pooling not in {"last", "mean", "max", "meanmax"}:
            raise ValueError("pooling must be last, mean, max or meanmax")
        if input_size not in {1, 300}:
            raise ValueError("input_size must be 300 (classroom) or 1 (pointwise sequence)")
        if not classifier_hidden_sizes or any(
                not isinstance(width, int) or width < 1 for width in classifier_hidden_sizes):
            raise ValueError("classifier_hidden_sizes must contain positive integers")
        self.model_type = model_type
        self.pooling = pooling
        self.input_size = input_size
        self.classifier_hidden_sizes = tuple(classifier_hidden_sizes)
        recurrent_cls = nn.LSTM if model_type == "lstm" else nn.RNN
        self.recurrent = recurrent_cls(input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=recurrent_dropout if num_layers > 1 else 0.0)
        feature_size = hidden_size * (2 if pooling == "meanmax" else 1)
        layers = []
        for width in self.classifier_hidden_sizes:
            layers.extend((nn.Linear(feature_size, width), nn.ReLU(),
                           nn.Dropout(classifier_dropout)))
            feature_size = width
        layers.append(nn.Linear(feature_size, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            raise ValueError(f"Expected [batch, time, {self.input_size}], received {tuple(x.shape)}")
        if self.input_size == 300 and x.shape[1] != 1:
            raise ValueError(f"Classroom input must be [batch, 1, 300], received {tuple(x.shape)}")
        sequence, state = self.recurrent(x)
        if self.pooling == "last":
            hidden = state[0] if self.model_type == "lstm" else state
            features = hidden[-1]
        elif self.pooling == "mean":
            features = sequence.mean(dim=1)
        elif self.pooling == "max":
            features = sequence.amax(dim=1)
        else:
            features = torch.cat((sequence.mean(dim=1), sequence.amax(dim=1)), dim=1)
        return self.classifier(features)


def build_model(config: dict) -> ECGClassifier:
    config = dict(config)
    # 修改前保存的配置没有 input_size，实际训练时均使用 1 维逐点输入。
    # 新训练必须显式保存 input_size=300，旧权重继续按原结构加载。
    config.setdefault("input_size", 1)
    # 历史模型只有一个 64 单元的分类器隐藏层；保留其层名和权重形状。
    # 新训练会明确保存所有结构参数，避免默认结构更新影响旧模型。
    config.setdefault("classifier_hidden_sizes", [64])
    config.setdefault("hidden_size", 64)
    config.setdefault("num_layers", 1)
    config.setdefault("recurrent_dropout", 0.0)
    config.setdefault("classifier_dropout", 0.3)
    return ECGClassifier(**config)
