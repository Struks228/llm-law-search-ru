# data.py
import os
import torch
from torch.utils.data import TensorDataset
from tokenizers import ByteLevelBPETokenizer
from config import Config

def load_tokenizer(cfg: Config):
    tok = ByteLevelBPETokenizer(
        os.path.join(cfg.tokenizer_dir, "vocab.json"),
        os.path.join(cfg.tokenizer_dir, "merges.txt"),
    )
    return tok

def create_dataset_from_ids(ids: torch.Tensor, block_size: int, step: int):
    X, Y = [], []
    for i in range(0, len(ids) - block_size - 1, step):
        X.append(ids[i:i+block_size])
        Y.append(ids[i+1:i+block_size+1])
    return TensorDataset(torch.stack(X), torch.stack(Y))


def prepare_datasets(cfg: Config):
    tok = load_tokenizer(cfg)
    print("Токенизатор загружен, vocab_size =", tok.get_vocab_size())

    with open(cfg.raw_corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    ids = torch.tensor(tok.encode(text).ids, dtype=torch.long)

    split = int(len(ids) * (1.0 - cfg.val_split))
    train_ids = ids[:split]
    val_ids   = ids[split:]

    # Сохраняем сырые токены, если захочешь пересобрать датасет
    torch.save(train_ids, os.path.join(cfg.dataset_dir, "train_ids.pt"))
    torch.save(val_ids,   os.path.join(cfg.dataset_dir, "val_ids.pt"))

    train_dataset = create_dataset_from_ids(train_ids, cfg.block_size, cfg.step)
    val_dataset   = create_dataset_from_ids(val_ids,   cfg.block_size, cfg.step)

    torch.save(train_dataset, os.path.join(cfg.dataset_dir, "train.pt"))
    torch.save(val_dataset,   os.path.join(cfg.dataset_dir, "val.pt"))

    print("Train windows:", len(train_dataset), "Val windows:", len(val_dataset))


if __name__ == "__main__":
    cfg = Config()
    os.makedirs(cfg.dataset_dir, exist_ok=True)
    prepare_datasets(cfg)
