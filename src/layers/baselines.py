import torch.nn as nn


class StandardFP32Linear(nn.Linear):
    """
    Обёртка над обычным nn.Linear.
    Игнорирует параметры group_size и latent_dim, чтобы быть совместимой
    с архитектурой HypercubeGPT (которая передаёт их в linear_cls).
    """
    def __init__(self, in_features: int, out_features: int,
                 group_size=None, latent_dim=None, bias=True, **kwargs):
        super().__init__(in_features, out_features, bias=bias)
