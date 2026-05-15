# not going to use ddp initially
import torch
import tiktoken
import math
import time

from .data.dataloader import DataLoaderLite
from .model.gpt_model import GPT
from .model.config import GPTConfig


ddp_rank = 0
ddp_local_rank = 0
ddp_world_size = 1
master_process = True

device = "cpu" # ?
if torch.cuda.is_available():
    device = "cuda"

print(f"using device {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

SEED = 1337
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288 # 2^19? Why? This is in number of tokens, TODO: different for shakespeare
B = 64 # Batch size
T = 1024 # Sequence length / Context window

assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"

grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision("high")

model = GPT(GPTConfig()) # TODO: Modify vocab to 50304 to make it a multiple of 32/64/128
model.to(device)

max_lr = 6e-4 
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # Depends on the dataset + batch size... TODO

def get_lr(it):

    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps

    if it > max_steps:
        return min_lr
    
    # cosine decay for learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# TODO: Logging, optimizer
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # Wait for gpu to finish work
    
    t1 = time.time()
    dt = t1 - t0

    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt

    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec: .2f}")
        


