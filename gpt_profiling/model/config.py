


class GPTConfig:
    vocab_size = None
    block_size = None
    n_layer = None
    n_head = None
    n_embd = None
    dropout = None
    bias = None
    def __init__(self, vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768, dropout=0.1, bias=True) -> None:
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias