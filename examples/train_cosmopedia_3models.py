import argparse
import csv
import json
import math
import os
import sys
import time

# Принудительно UTF-8 в stdout/stderr (эмодзи и кириллица под Windows cp1251)
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformers import HypercubeGPT
from src.layers.hypercube_linear_layers import (
    BaseHypercubeLinear,
    TernaryHypercubeLinear,
)
from src.layers.ste_linear_layers import (
    QuantizedLinearSTE,
    TwoBitSTELinear,
)
from src.layers.baselines import StandardFP32Linear

# ==========================================
# ПЛАН ЭКСПЕРИМЕНТА
# ==========================================
# Масштабирование с TinyStories на Cosmopedia (char-level, ~50M токенов).
# Сравниваем трёх «лучших представителей» семейств из прошлого эксперимента:
#   1. FP32                       — верхняя граница качества
#   2. Ternary Hypercube          — стабильный гиперкуб (латентное сжатие + STE)
#   3. TwoBit STE                 — точка Парето прошлого прогона (плотный STE)


# ==========================================
# КОНФИГУРАЦИЯ
# ==========================================
SEED = 1337
BLOCK_SIZE = 256
BATCH_SIZE = 32
MAX_ITERS = 610          # ~1 эпоха по 50M char-токенов (50M / (32*256) ≈ 6104)
EVAL_INTERVAL = 500
EVAL_ITERS = 100
LR = 1e-3
GRAD_CLIP = 1.0           # защита от взрыва градиентов на длинном прогоне

EMBED_DIM = 384
HEADS = 6
DEPTH = 6
LATENT_DIM = 3            # для гиперкуба

# Объём сабсета Cosmopedia в символах (char-level: 1 символ ≈ 1 токен).
COSMO_SUBSET_CHARS = 50_000_000
COSMO_CONFIG = "stories"  # конфиг HuggingFaceTB/cosmopedia

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ (Cosmopedia)
# ==========================================
def load_cosmopedia_subset(n_chars: int, config: str = COSMO_CONFIG,
                           cache_path: str = None):
    """
    Стримит датасет HuggingFaceTB/cosmopedia (указанный config) и накапливает
    первые n_chars символов поля `text`. Кэширует результат в файл, имя
    которого включает размер, чтобы разные сабсеты не конфликтовали.
    """
    if cache_path is None:
        cache_path = f"cosmopedia_{config}_subset_{n_chars}.txt"
    if os.path.exists(cache_path):
        print(f"📂 Загружаю кэшированный сабсет Cosmopedia из {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()

    print(f"⏳ Стримлю Cosmopedia (config={config}) до {n_chars:,} символов...")
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceTB/cosmopedia", config, split="train",
                      streaming=True)

    chunks = []
    total = 0
    ndocs = 0
    for ex in ds:
        text = ex.get("text", "")
        if not text:
            continue
        text = text.strip() + "\n"
        chunks.append(text)
        total += len(text)
        ndocs += 1
        if ndocs % 5000 == 0:
            print(f"   ... {ndocs} док., {total:,} символов")
        if total >= n_chars:
            break

    full = "".join(chunks)[:n_chars]
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(full)
    print(f"✅ Сохранён сабсет ({len(full):,} символов, {ndocs} док.) в {cache_path}")
    return full


def build_dataset(text: str):
    """Символьная токенизация."""
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    return train_data, val_data, vocab_size, encode, decode


TRAIN_DATA = None
VAL_DATA = None


def get_batch(split):
    data_split = TRAIN_DATA if split == 'train' else VAL_DATA
    ix = torch.randint(len(data_split) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_split[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data_split[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, split='val'):
    model.eval()
    losses = torch.zeros(EVAL_ITERS)
    for k in range(EVAL_ITERS):
        X, Y = get_batch(split)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


# ==========================================
# ОПИСАНИЕ МОДЕЛЕЙ
# ==========================================
def model_configs(vocab_size: int):
    cfg = dict(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        embed_dim=EMBED_DIM,
        heads=HEADS,
        depth=DEPTH,
    )

    def build(linear_cls, group_size, latent_dim):
        return HypercubeGPT(
            linear_cls=linear_cls,
            group_size=group_size,
            latent_dim=latent_dim,
            **cfg,
        )

    return [
        #("TwoBit_STE", lambda: build(TwoBitSTELinear, None, None)),
        #("FP32", lambda: build(StandardFP32Linear, None, None)),
        ("Ternary_Hypercube", lambda: build(TernaryHypercubeLinear, 5, LATENT_DIM)),
    ]


# ==========================================
# ПОДСЧЁТ ПАРАМЕТРОВ / СТОИМОСТИ ХРАНЕНИЯ
# ==========================================
@torch.no_grad()
def count_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def model_weight_bits(model: nn.Module):
    """См. train_tinystories_7models.py — идентичная логика оценки бит/вес."""
    quant_bits = {
        TwoBitSTELinear: 2,
        TernaryHypercubeLinear: None,
    }

    total_weight_elems = 0
    total_storage_bits = 0

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not isinstance(mod, QuantizedLinearSTE):
            n = mod.weight.numel()
            total_weight_elems += n
            total_storage_bits += n * 32
        elif isinstance(mod, QuantizedLinearSTE):
            n = mod.weight.numel()
            bits = quant_bits.get(type(mod), 32)
            total_weight_elems += n
            total_storage_bits += n * bits
        elif isinstance(mod, BaseHypercubeLinear):
            latent_elems = mod.weight_latent.numel() + mod.proj_matrix.numel()
            eff_weights = mod.out_features * mod.pad_features
            total_weight_elems += eff_weights
            total_storage_bits += latent_elems * 32

    avg_bits_per_weight = total_storage_bits / max(total_weight_elems, 1)
    return total_weight_elems, total_storage_bits, avg_bits_per_weight


# ==========================================
# ЦИКЛ ОБУЧЕНИЯ
# ==========================================
def train_one_model(name, build_fn, decode, run_dir):
    set_seed(SEED)
    print(f"\n{'=' * 60}")
    print(f"🚀 Модель: {name}")
    print(f"{'=' * 60}")

    model = build_fn().to(device)
    n_params = count_params(model)
    n_elems, storage_bits, bpw = model_weight_bits(model)
    print(f"Параметров: {n_params:,} | весовых элементов: {n_elems:,} | "
          f"среднее хранение: {bpw:.2f} бит/вес | ~{storage_bits / 8 / 1e6:.3f} МБ на веса")

    optimizer = optim.AdamW(model.parameters(), lr=LR)

    history = []
    t0 = time.time()
    best_val = float('inf')

    for it in tqdm(range(MAX_ITERS)):
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            tr = estimate_loss(model, 'train')
            va = estimate_loss(model, 'val')
            best_val = min(best_val, va)
            history.append({"iter": it, "train_loss": tr, "val_loss": va})
            elapsed = time.time() - t0
            print(f"  шаг {it:5d} | train {tr:.4f} | val {va:.4f} | "
                  f"best {best_val:.4f} | {elapsed:.0f}с")

        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if GRAD_CLIP is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

    dt = time.time() - t0
    final_val = history[-1]["val_loss"]
    print(f"⏱  Время обучения: {dt:.1f} с | Финальный val loss: {final_val:.4f}")

    # Генерация сэмплов
    model.eval()
    samples = []
    with torch.no_grad():
        for _ in range(2):
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            tokens = model.generate(context, max_new_tokens=160, temperature=1.0)[0].tolist()
            samples.append(decode(tokens))

    result = {
        "name": name,
        "n_params": n_params,
        "weight_elems": n_elems,
        "avg_bits_per_weight": bpw,
        "storage_mb": storage_bits / 8 / 1e6,
        "final_val_loss": final_val,
        "best_val_loss": best_val,
        "train_time_sec": dt,
        "history": history,
        "samples": samples,
    }

    with open(os.path.join(run_dir, f"{name}_result.json"), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ==========================================
# MAIN
# ==========================================
def main():
    global MAX_ITERS, TRAIN_DATA, VAL_DATA

    parser = argparse.ArgumentParser(description="Cosmopedia: 3-model comparison")
    parser.add_argument("--subset_chars", type=int, default=COSMO_SUBSET_CHARS,
                        help="Размер сабсета Cosmopedia в символах (char-токенах)")
    parser.add_argument("--config", type=str, default=COSMO_CONFIG,
                        help="config cosmopedia (stories / web_samples_v1 / ...)")
    parser.add_argument("--iters", type=int, default=MAX_ITERS)
    parser.add_argument("--out", type=str, default="runs/cosmopedia_3models",
                        help="Каталог для результатов")
    args = parser.parse_args()

    MAX_ITERS = args.iters

    set_seed(SEED)
    os.makedirs(args.out, exist_ok=True)

    print(f"🖥  Устройство: {device.upper()}")

    text = load_cosmopedia_subset(args.subset_chars, args.config)
    train_data, val_data, vocab_size, encode, decode = build_dataset(text)
    TRAIN_DATA, VAL_DATA = train_data, val_data
    print(f"Vocab size: {vocab_size} | train tokens: {len(train_data):,} | "
          f"val tokens: {len(val_data):,}")
    print(f"Итераций: {MAX_ITERS} | эпоха ~ {len(train_data)//(BATCH_SIZE*BLOCK_SIZE)}")

    configs = model_configs(vocab_size)
    print(f"\nЗапускаю {len(configs)} моделей по {MAX_ITERS} итераций каждая...")

    results = []
    for name, build_fn in configs:
        try:
            res = train_one_model(name, build_fn, decode, args.out)
            results.append(res)
        except Exception as e:
            print(f"❌ Ошибка при обучении {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"name": name, "error": str(e)})

    # ===== Сводная таблица =====
    print("\n" + "=" * 90)
    print(f"{'Модель':<22}{'Параметры':>12}{'бит/вес':>10}{'val loss':>11}{'время(с)':>10}")
    print("=" * 90)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<22}{'ERROR':>12}{'-':>10}{'-':>11}{'-':>10}")
            continue
        print(f"{r['name']:<22}{r['n_params']:>12,}{r['avg_bits_per_weight']:>10.2f}"
              f"{r['final_val_loss']:>11.4f}{r['train_time_sec']:>10.1f}")
    print("=" * 90)

    # CSV сводка
    csv_path = os.path.join(args.out, "summary.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["model", "n_params", "weight_elems", "avg_bits_per_weight",
                    "storage_mb", "final_val_loss", "best_val_loss", "train_time_sec"])
        for r in results:
            if "error" in r:
                w.writerow([r["name"], "ERROR"] + [""] * 7)
                continue
            w.writerow([r["name"], r["n_params"], r["weight_elems"],
                        f"{r['avg_bits_per_weight']:.4f}", f"{r['storage_mb']:.4f}",
                        f"{r['final_val_loss']:.4f}", f"{r['best_val_loss']:.4f}",
                        f"{r['train_time_sec']:.1f}"])
    print(f"\n💾 Результаты сохранены в {args.out}/ (summary.csv + <model>_result.json)")


if __name__ == "__main__":
    main()
