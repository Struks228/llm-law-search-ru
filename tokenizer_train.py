# tokenizer_train.py
import os
from tokenizers import ByteLevelBPETokenizer
from config import Config

if __name__ == "__main__":
    cfg = Config()

    assert os.path.exists(cfg.raw_corpus_path), \
        f"Не найден {cfg.raw_corpus_path}. Положи туда свой корпус raw.txt"

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=[cfg.raw_corpus_path],
        vocab_size=cfg.vocab_size,
        min_frequency=2,
        special_tokens=[
            "<s>", "<pad>", "</s>", "<unk>", "<mask>"
        ],
    )

    tokenizer.save_model(cfg.tokenizer_dir)
    print("Токенизатор обучен и сохранён в", cfg.tokenizer_dir)
