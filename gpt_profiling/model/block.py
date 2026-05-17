import torch
import torch.nn as nn
import torch.functional as F
from .causal_self_attention import CausalSelfAttention
from .mlp import MLP

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x)) # relu?
        x = x + self.mlp(self.ln2(x))

        return x