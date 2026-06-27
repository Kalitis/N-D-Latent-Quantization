import os
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim

from src.models import HypercubeGPT
from src.layers import TernaryHypercubeLinear

# ==========================================
# АДАПТЕР ДЛЯ СТАНДАРТНОГО FP32 СЛОЯ
# ==========================================
class StandardFP32Linear(nn.Linear):
    """
    Обертка над обычным nn.Linear. 
    Игнорирует параметры group_size и latent_dim, чтобы совместиться с архитектурой HypercubeGPT.
    """
    def __init__(self, in_features: int, out_features: int, group_size=None, latent_dim=None, bias=True, **kwargs):
        super().__init__(in_features, out_features, bias=bias)

# ==========================================
# КОНФИГУРАЦИЯ ДАТАСЕТА (Упрощенно)
# ==========================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 128
batch_size = 64
max_iters = 1000  # Сделаем короче, так как учим две модели подряд

data_path = 'input.txt'
if not os.path.exists(data_path):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", data_path)

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
encode = lambda s: [{ch: i for i, ch in enumerate(chars)}[c] for c in s]
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = torch.zeros(100)
    for k in range(100):
        X, Y = get_batch('val')
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()

# ==========================================
# ФУНКЦИЯ ОБУЧЕНИЯ ДЛЯ СРАВНЕНИЯ
# ==========================================
def run_experiment(name, LayerClass, group_size=None):
    print(f"\n🚀 Обучение Baseline: {name}")
    model = HypercubeGPT(
        vocab_size=vocab_size, block_size=block_size, embed_dim=128, 
        heads=4, depth=4, linear_cls=LayerClass, group_size=group_size, latent_dim=3
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    for iter in range(max_iters):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if (iter + 1) % 500 == 0:
            print(f"Шаг {iter+1:4d} завершен...")
            
    final_val_loss = estimate_loss(model)
    return final_val_loss

if __name__ == "__main__":
    results = {}
    
    # 1. Запускаем стандартный плотный FP32 трансформер
    results["Standard FP32"] = run_experiment("Standard FP32", StandardFP32Linear)
    
    # 2. Запускаем нашу кастомную тернарную архитектуру
    results["Ternary Hypercube (Group 5)"] = run_experiment("Ternary Hypercube (Group 5)", TernaryHypercubeLinear, group_size=5)
    
    print("\n" + "="*50)
    print("🏆 ИТОГОВОЕ СРАВНЕНИЕ (Validation Loss)")
    print("="*50)
    for name, loss in results.items():
        print(f"{name:<30} | Loss: {loss:.4f}")
    print("="*50)
    print("Примечание: Меньший Loss означает лучшее качество.")