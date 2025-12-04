# config.py
import os
import torch

class Config:
    # Пути
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir     = os.path.join(project_root, "data")
    tokenizer_dir = os.path.join(data_dir, "tokenizer")
    dataset_dir   = os.path.join(data_dir, "dataset")
    ckpt_dir      = os.path.join(project_root, "checkpoints")

    raw_corpus_path = os.path.join(data_dir, "raw.txt")
    final_model_path = os.path.join(ckpt_dir, "minigpt2_final.pt")

    os.makedirs(tokenizer_dir, exist_ok=True)
    os.makedirs(dataset_dir,  exist_ok=True)
    os.makedirs(ckpt_dir,     exist_ok=True)

    # Токенизатор
    vocab_size = 8000

    # Модель
    block_size = 128   # больше контекст, чем 128
    n_embd     = 128
    n_layer    = 4
    n_head     = 4
    dropout    = 0.1

    # Обучение
    batch_size = 64
    epochs     = 20
    early_stopping_patience = 2
    lr         = 1e-4
    step       = 32      # шаг окна по датасету (stride)
    val_split  = 0.1

    # Устройство
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = (device == "cuda")
