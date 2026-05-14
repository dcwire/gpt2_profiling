import os
import requests
import tiktoken
import numpy as np
import pathlib

def load_tinyshakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text

    n = len(text)
    train = text[:int(0.9 * n)]
    val = text[int(0.9 * n): ]

    path = os.path.join(pathlib.Path(__file__).parent.resolve(), "tinyshakespeare") 
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train)
    val_ids = enc.encode_ordinary(val)

    os.makedirs(path, exist_ok=True)
    train_ids_np = np.array(train_ids, dtype=np.uint16)
    val_ids_np = np.array(val_ids, dtype=np.uint16)

    train_ids_np.tofile(os.path.join(path, "train_000.bin"))
    val_ids_np.tofile(os.path.join(path, "val_000.bin"))
    
    return True

