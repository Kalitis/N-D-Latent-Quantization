import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.hypercube_linear_layers import BinarySTE, TernarySTE, TwoBitSTE


# ==========================================
# ПЛОТНЫЕ СЛОИ С ПРЯМЫМ STE-КВАНТОВАНИЕМ (BitNet-style)
# ==========================================
# В отличие от гиперкубических слоев, здесь НЕТ латентной параметризации:
# веса хранятся плотно (как в nn.Linear) и квантуются через STE на прямом
# проходе. Это чистый baseline квантованного обучения без сжатия латентным
# подпространством. Сигнатура конструктора совместима с HypercubeGPT
# (принимает и игнорирует group_size / latent_dim).


class QuantizedLinearSTE(nn.Module):
    """
    Базовый плотный линейный слой с квантованием весов через STE.
    Нормализация весов повторяет канон BitNet (как в BaseHypercubeLinear):
    построчное центрирование + глобальное масштабирование по среднему модулю.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size=None,
        latent_dim=None,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.weight.size(1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Субклассы должны реализовать метод quantize()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        # BitNet-style нормализация (идентична BaseHypercubeLinear)
        w_centered = w - w.mean(dim=-1, keepdim=True)
        scale = w_centered.abs().mean() + 1e-5
        w_normalized = w_centered / scale

        w_quantized = self.quantize(w_normalized)
        w_effective = w_quantized * scale

        return F.linear(x, w_effective, self.bias)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(in_features={self.in_features}, "
                f"out_features={self.out_features}, bias={self.bias is not None})")


class BinarySTELinear(QuantizedLinearSTE):
    """Плотный слой с бинарным квантованием {-1, 1} через STE."""
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        return BinarySTE.apply(x)


class TernarySTELinear(QuantizedLinearSTE):
    """Плотный слой с тернарным квантованием {-1, 0, 1} через STE."""
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        return TernarySTE.apply(x)


class TwoBitSTELinear(QuantizedLinearSTE):
    """Плотный слой с 2-битным квантованием {-2, -1, 0, 1} через STE."""
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        return TwoBitSTE.apply(x)
