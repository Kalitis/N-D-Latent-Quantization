import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ==========================================
# 1. СТРАЙТ-ФРУ ЭСТИМАТОРЫ (STE)
# ==========================================

class BinarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Квантование в {-1, 1}
        return torch.sign(x) + (x == 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Квантование в {-1, 0, 1}
        return torch.round(torch.clamp(x, -1.0, 1.0))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class TwoBitSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 2 бита = 4 состояния. Разумно взять симметричные уровни {-2, -1, 1, 2} 
        # или равномерные {-2, -1, 0, 1}. Сделаем b1.58-style расширение: {-2, -1, 0, 1}
        return torch.round(torch.clamp(x, -2.0, 1.0))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# ==========================================
# 2. БАЗОВЫЙ ГИПЕРКУБИЧЕСКИЙ СЛОЙ
# ==========================================

class BaseHypercubeLinear(nn.Module):
    """
    Базовый класс для линейных слоев с низкоразмерной параметризацией весов.
    Проецирует латентное подпространство в N-мерные группы и применяет квантование.
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        group_size: int, 
        latent_dim: int, 
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.latent_dim = latent_dim
        
        # Автоматический паддинг входа до кратности размеру группы
        self.pad_features = group_size * ((in_features + (group_size - 1)) // group_size)
        self.padding_size = self.pad_features - in_features
        self.groups = self.pad_features // group_size
        
        # Непрерывные латентные координаты весов [out_features, groups, latent_dim]
        self.weight_latent = nn.Parameter(torch.randn(out_features, self.groups, latent_dim) * 0.02)
        
        # Обучаемая матрица проекции [latent_dim, group_size]
        raw_proj = torch.randn(latent_dim, group_size)
        self.proj_matrix = nn.Parameter(F.normalize(raw_proj, p=2, dim=0))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Абстрактный метод для применения конкретного STE алгоритма."""
        raise NotImplementedError("Субклассы должны реализовать метод quantize()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Динамический паддинг входа при необходимости
        if self.padding_size > 0:
            x = F.pad(x, (0, self.padding_size), "constant", 0)
            
        # 2. Проекция латентных координат в непрерывное пространство группы
        # [out, groups, latent_dim] @ [latent_dim, group_size] -> [out, groups, group_size]
        W_continuous = torch.matmul(self.weight_latent, self.proj_matrix)
        W_continuous = W_continuous.view(self.out_features, self.pad_features)
        
        # 3. Стандартизация (Центрирование и Масштабирование) по канонам BitNet
        W_centered = W_continuous - W_continuous.mean(dim=-1, keepdim=True)
        scale = W_centered.abs().mean(dim=-1, keepdim=True) + 1e-5
        W_normalized = W_centered / scale
        
        # 4. Квантование через переопределенный в субклассе STE
        W_quantized = self.quantize(W_normalized)
        
        # 5. Де-масштабирование для сохранения дисперсии активаций
        W_effective = W_quantized * scale
        
        # 6. Классический линейный трансформ
        return F.linear(x, W_effective, self.bias)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(in_features={self.in_features}, "
                f"out_features={self.out_features}, group_size={self.group_size}, "
                f"latent_dim={self.latent_dim}, bias={self.bias is not None})")


# ==========================================
# 3. КОНКРЕТНЫЕ РЕАЛИЗАЦИИ СЛОЕВ
# ==========================================

class BinaryHypercubeLinear(BaseHypercubeLinear):
    """Бинарный гиперкубический слой. По умолчанию сжимает 8 весов в 3 латентных параметра."""
    def __init__(self, in_features: int, out_features: int, group_size: int = 8, latent_dim: int = 3, bias: bool = True):
        super().__init__(in_features, out_features, group_size, latent_dim, bias)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        return BinarySTE.apply(x)


class TernaryHypercubeLinear(BaseHypercubeLinear):
    """Тернарный гиперкубический слой. По умолчанию сжимает 5 весов в 3 латентных параметра."""
    def __init__(self, in_features: int, out_features: int, group_size: int = 5, latent_dim: int = 3, bias: bool = True):
        super().__init__(in_features, out_features, group_size, latent_dim, bias)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        return TernarySTE.apply(x)


class TwoBitHypercubeLinear(BaseHypercubeLinear):
    """2-битный (4 уровня) гиперкубический слой. По умолчанию сжимает 5 весов в 3 латентных параметра."""
    def __init__(self, in_features: int, out_features: int, group_size: int = 5, latent_dim: int = 3, bias: bool = True):
        super().__init__(in_features, out_features, group_size, latent_dim, bias)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        return TwoBitSTE.apply(x)