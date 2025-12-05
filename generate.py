# generate.py
import os
import torch
import torch.nn.functional as F

from config import Config
from model import MiniGPT2
from data import load_tokenizer

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1)
    return torch.where(logits < min_values, torch.full_like(logits, -1e10), logits)

@torch.no_grad()
def generate(model, tokenizer, prompt, cfg: Config, max_new_tokens=60, temperature=0.5, top_k=20):
    model.eval()
    tokens = tokenizer.encode(prompt).ids
    x = torch.tensor(tokens, dtype=torch.long, device=cfg.device).unsqueeze(0)

    for _ in range(max_new_tokens):
        x_cond = x[:, -cfg.block_size:] if x.size(1) > cfg.block_size else x
        logits = model(x_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-9)
        logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(x[0].tolist())


def main():
    cfg = Config()
    print("device =", cfg.device)

    tok = load_tokenizer(cfg)

    model = MiniGPT2(
        vocab_size=cfg.vocab_size,
        block_size=cfg.block_size,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        dropout=cfg.dropout
    ).to(cfg.device)

    if not os.path.exists(cfg.final_model_path):
        raise FileNotFoundError(f"Не найден {cfg.final_model_path}. Сначала запусти train.py")

    model.load_state_dict(torch.load(cfg.final_model_path, map_location=cfg.device))
    print("Модель загружена.")

    while True:
        prompt = input("\nВведите запрос (или пустую строку для выхода): ")
        if not prompt.strip():
            break
        generated = generate(model, tok, prompt, cfg, max_new_tokens=200, temperature=1.0, top_k=40)
        print("\n--- GENERATED ---\n")
        print(generated)


if __name__ == "__main__":
    main()
