import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Optional, Tuple
from src.layers import BaseHypercubeLinear, TernaryHypercubeLinear


class CausalSelfAttention(nn.Module):
    """
    Блок каузального Multi-Head Self-Attention.
    Использует кастомные гиперкубические слои для всех проекций.
    """
    def __init__(
        self, 
        embed_dim: int, 
        heads: int, 
        linear_cls: Type[BaseHypercubeLinear],
        **linear_kwargs
    ):
        super().__init__()
        assert embed_dim % heads == 0, "embed_dim должен делиться на количество голов (heads)"
        
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        
        # Проекции Query, Key, Value и Output
        self.q_proj = linear_cls(embed_dim, embed_dim, **linear_kwargs)
        self.k_proj = linear_cls(embed_dim, embed_dim, **linear_kwargs)
        self.v_proj = linear_cls(embed_dim, embed_dim, **linear_kwargs)
        self.o_proj = linear_cls(embed_dim, embed_dim, **linear_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (embed_dim)
        
        # Проекция и изменение формы для Multi-Head вычислений
        # [B, T, C] -> [B, T, heads, head_dim] -> [B, heads, T, head_dim]
        q = self.q_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # Вычисление матриц внимания (Scaled Dot-Product Attention)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        
        # Каузальная маска, чтобы заблокировать взгляд в будущее
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        # Объединение голов обратно в один вектор
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        
        return self.o_proj(y)


class TransformerBlock(nn.Module):
    """
    Классический блок Трансформера (Pre-LN архитектура).
    Объединяет CausalSelfAttention и Feed-Forward Network (MLP).
    """
    def __init__(
        self, 
        embed_dim: int, 
        heads: int, 
        linear_cls: Type[BaseHypercubeLinear],
        **linear_kwargs
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, heads, linear_cls, **linear_kwargs)
        
        self.ln_2 = nn.LayerNorm(embed_dim)
        # Полносвязная сеть (FFN / MLP) с расширением скрытого слоя в 4 раза
        self.mlp = nn.Sequential(
            linear_cls(embed_dim, 4 * embed_dim, **linear_kwargs),
            nn.GELU(),
            linear_cls(4 * embed_dim, embed_dim, **linear_kwargs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm и реозидуальные связи (Residual Connections)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class HypercubeGPT(nn.Module):
    """
    Полноценная декодерная языковая модель (GPT-style),
    построенная на базе квантованных гиперкубических слоев.
    """
    def __init__(
        self, 
        vocab_size: int, 
        block_size: int = 128, 
        embed_dim: int = 128, 
        heads: int = 4, 
        depth: int = 4,
        linear_cls: Type[BaseHypercubeLinear] = TernaryHypercubeLinear,
        **linear_kwargs
    ):
        super().__init__()
        self.block_size = block_size
        
        # Стандартные непрерывные эмбеддинги токенов и позиций
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        
        # Стек трансформер-блоков
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, heads, linear_cls, **linear_kwargs)
            for _ in range(depth)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # Выходная голова классификатора — тоже квантованная
        self.lm_head = linear_cls(embed_dim, vocab_size, **linear_kwargs)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        assert T <= self.block_size, f"Длина контекста {T} превышает максимально допустимую {self.block_size}"
        
        # Извлекаем позиционные и токенные эмбеддинги
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        # Прогоняем через все блоки трансформера
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Если переданы таргеты, сразу считаем Cross-Entropy Loss (удобно для обучения)
        loss = None
        if targets is not None:
            B, T, C = logits.size()
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """Авторегрессионная генерация текста токен за токеном."""
        self.eval()
        for _ in range(max_new_tokens):
            # Обрезаем контекст до максимального размера block_size
            idx_cond = idx[:, -self.block_size:]
            
            logits, _ = self(idx_cond)
            # Берем логиты только для самого последнего временного шага
            logits = logits[:, -1, :] / (temperature + 1e-8)
            
            probs = F.softmax(logits, dim=-1)
            # Сэмплируем следующий токен из распределения вероятностей
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx