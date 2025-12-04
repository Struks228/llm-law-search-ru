# model.py
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            batch_first=True
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(attn_out)
        h = self.ln2(x)
        x = x + self.dropout(self.mlp(h))
        return x


class MiniGPT2(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_layer, n_head, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb   = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, dropout=dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight,      std=0.02)

    def forward(self, idx):
        B, T = idx.size()
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.block_size}")

        tok = self.token_emb(idx)  # (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device)).unsqueeze(0)  # (1, T, C)
        x = tok + pos

        # causal mask
        mask = torch.triu(
            torch.ones(T, T, device=idx.device, dtype=torch.bool),
            diagonal=1
        )
        attn_mask = torch.zeros((T, T), device=idx.device, dtype=torch.float32)
        attn_mask = attn_mask.masked_fill(mask, float('-inf'))

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
