# train.py

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.multiprocessing import freeze_support

from config import Config
from model import MiniGPT2
from data import load_tokenizer  # если не используешь, можно удалить


def load_datasets(cfg: Config):
    """
    Загружаем уже подготовленные датасеты train.pt и val.pt.
    Ожидается, что это torch.utils.data.Dataset (например, TensorDataset).
    """
    train_path = os.path.join(cfg.dataset_dir, "train.pt")
    val_path   = os.path.join(cfg.dataset_dir, "val.pt")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Не найден train датасет: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Не найден val датасет: {val_path}")

    train_dataset = torch.load(train_path)
    val_dataset   = torch.load(val_path)

    return train_dataset, val_dataset


def evaluate(model: MiniGPT2, data_loader: DataLoader, cfg: Config) -> float:
    """
    Оцениваем модель на датасете data_loader.
    Возвращаем средний loss на токен.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(cfg.device)
            y_batch = y_batch.to(cfg.device)

            logits = model(x_batch)  # (B, T, vocab)

            loss = F.cross_entropy(
                logits.view(-1, cfg.vocab_size),
                y_batch.view(-1),
                reduction="sum"  # аккумулируем сумму, потом поделим на число токенов
            )

            total_loss += loss.item()
            total_tokens += y_batch.numel()

    if total_tokens == 0:
        return float("inf")

    return total_loss / total_tokens


def main():
    freeze_support()  # нужно для Windows при использовании DataLoader

    cfg = Config()
    print("device =", cfg.device)

    # ====== Датасеты ======
    train_dataset, val_dataset = load_datasets(cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,  # для Windows безопасно
        pin_memory=(cfg.device == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(cfg.device == "cuda")
    )

    # ====== Модель ======
    model = MiniGPT2(
        vocab_size=cfg.vocab_size,
        block_size=cfg.block_size,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        dropout=cfg.dropout,
    ).to(cfg.device)

    # num_params может быть либо атрибутом, либо считаем вручную
    if hasattr(model, "num_params"):
        print("Model params:", model.num_params / 1e6, "M")
    else:
        print("Model params:",
              sum(p.numel() for p in model.parameters()) / 1e6,
              "M")

    # ====== Оптимизатор и AMP ======
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp_enabled)

    best_val_loss = float("inf")

    # опциональный early stopping, если добавишь в Config
    patience = getattr(cfg, "early_stopping_patience", None)
    epochs_without_improvement = 0

    for epoch in range(cfg.epochs):
        model.train()
        train_loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{cfg.epochs}"
        )

        running_loss = 0.0  # сглаженный train loss
        total_train_loss = 0.0
        total_train_tokens = 0

        for x_batch, y_batch in train_loop:
            x_batch = x_batch.to(cfg.device)
            y_batch = y_batch.to(cfg.device)

            optimizer.zero_grad(set_to_none=True)

            # AMP (да, тут deprecated предупреждение, но оно не ломает код)
            with torch.cuda.amp.autocast(enabled=cfg.amp_enabled):
                logits = model(x_batch)
                loss = F.cross_entropy(
                    logits.view(-1, cfg.vocab_size),
                    y_batch.view(-1)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_item = loss.item()

            running_loss = (
                0.99 * running_loss + 0.01 * loss_item
                if running_loss != 0.0 else loss_item
            )

            total_train_loss += loss_item * y_batch.numel()
            total_train_tokens += y_batch.numel()

            train_loop.set_postfix(loss=loss_item, avg_loss=running_loss)

        train_epoch_loss = (
            total_train_loss / total_train_tokens
            if total_train_tokens > 0 else float("inf")
        )

        # ====== Валидация ======
        val_loss = evaluate(model, val_loader, cfg)
        print(f"\nEpoch {epoch + 1}: "
              f"train_loss = {train_epoch_loss:.4f}, "
              f"val_loss = {val_loss:.4f}, "
              f"train_avg_smooth = {running_loss:.4f}")

        # ====== Сохранение чекпоинта эпохи ======
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(cfg.ckpt_dir, f"minigpt2_epoch{epoch + 1}.pt")

        # вместо asdict(cfg) — берём обычный __dict__
        cfg_dict = getattr(cfg, "__dict__", None)
        if cfg_dict is None:
            cfg_dict = {}

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_epoch_loss,
                "config": cfg_dict,
            },
            ckpt_path,
        )
        print("Сохранён чекпоинт:", ckpt_path)

        # ====== Лучшая модель по val_loss ======
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(cfg.final_model_path), exist_ok=True)
            torch.save(model.state_dict(), cfg.final_model_path)
            epochs_without_improvement = 0
            print("Обновлена лучшая модель:", cfg.final_model_path)
        else:
            epochs_without_improvement += 1

        # ====== Ранняя остановка (если задана) ======
        if patience is not None and epochs_without_improvement >= patience:
            print(
                f"Ранняя остановка: не было улучшений {epochs_without_improvement} эпох."
            )
            break

    print("Обучение завершено. Лучшая val_loss:", best_val_loss)


if __name__ == "__main__":
    main()
