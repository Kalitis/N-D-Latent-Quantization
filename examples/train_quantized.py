import os
import sys
import urllib.request
import torch
import torch.optim as optim

# Импортируем нашу архитектуру из src
sys.path.append(os.path.abspath('..'))
from src.models.transformers import HypercubeGPT
from src.layers.hypercube_linear_layers import BinaryHypercubeLinear, TernaryHypercubeLinear, TwoBitHypercubeLinear

# ==========================================
# КОНФИГУРАЦИЯ И ЗАГРУЗКА ДАННЫХ
# ==========================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 128
batch_size = 64
max_iters = 2500
eval_interval = 250
learning_rate = 1e-3

# Настройки архитектуры
embed_dim = 128
heads = 4
depth = 4

print("⏳ Подготовка датасета TinyShakespeare...")
data_path = 'input.txt'
if not os.path.exists(data_path):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, data_path)

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

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
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(100)
        for k in range(100):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ==========================================
# ВЫБОР СЛОЯ И ОБУЧЕНИЕ
# ==========================================
if __name__ == "__main__":
    print(f"🚀 Запуск на устройстве: {device.upper()}")
    
    # Легкое переключение между режимами!
    # Варианты: BinaryHypercubeLinear, TernaryHypercubeLinear, TwoBitHypercubeLinear
    LayerClass = TwoBitHypercubeLinear
    group_size = 5  # Оптимальный размер для тернарного режима
    
    print(f"Создание модели с использованием: {LayerClass.__name__} (Group Size: {group_size})")
    
    model = HypercubeGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        embed_dim=embed_dim,
        heads=heads,
        depth=depth,
        linear_cls=LayerClass,
        group_size=group_size,
        latent_dim=3
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    print("\nНачинаем обучение...")
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"Шаг {iter:4d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    print("\n📝 Тестовая генерация:")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = model.generate(context, max_new_tokens=200)[0].tolist()
    print("-" * 50)
    print(decode(generated_tokens))
    print("-" * 50)