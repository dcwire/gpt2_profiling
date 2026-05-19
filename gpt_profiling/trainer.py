# not going to use ddp initially
import torch
import torch.nn.functional as F
import tiktoken
import math
import time
import wandb
import os
import pathlib
from dotenv import load_dotenv
from torch.profiler import profile, tensorboard_trace_handler, schedule, ProfilerActivity

from data.dataloader import DataLoaderLite
from model.gpt_model import GPT
from model.config import GPTConfig

parent_path = pathlib.Path(__file__).parent
env_path = os.path.join(parent_path.parent.resolve(), ".env")
if not os.path.exists(env_path):
    print("can't find .env file")
else:
    load_dotenv(env_path)

def train_gpt(checkpoint_path=None, profile_run=False):
    
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

    total_batch_size = 8192 # 2^19? Why? This is in number of tokens, TODO: different for shakespeare
    B = 8 # Batch size
    T = 1024 # Sequence length / Context window

    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"

    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    dataset = "tinyshakespeare"
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", data_root=dataset)
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", data_root=dataset)

    torch.set_float32_matmul_precision("high")

    gpt_config = GPTConfig()
    model = GPT(gpt_config) # TODO: Modify vocab to 50304 to make it a multiple of 32/64/128
    
    max_epoch = -1
    if (checkpoint_path is not None):
        try:
            path = os.path.join(parent_path.resolve(), checkpoint_path)
            if (os.path.exists(path)):
                # Find model with highest epoch
                files = os.listdir(path)
                
                for file in files:
                    if "gpt2_epoch" in file:
                        try:
                            epoch_num = int(file.split("gpt2_epoch_")[1].split('.')[0])
                            if (max_epoch == -1):
                                max_epoch = epoch_num
                            else:
                                max_epoch = max(max_epoch, epoch_num)
                        except Exception as e:
                            raise Exception(f"An exception occurred while loading model weights: {e}")
                if max_epoch != -1:
                    model.load_state_dict(torch.load(os.path.join(path, f"gpt2_epoch_{max_epoch}.pth"), weights_only=True))
                else:
                    print("unable to get the proper file for loading weights")
            else:
                print(f"checkpoint path not found at {path}. creating new path")
                os.makedirs(path, exist_ok=True)
        except Exception as e:
            print("an exception occurred when loading model, starting with max_epoch 0: ", e)
    
    if (max_epoch < 0):
        max_epoch = 0

    model.to(device)
    
    max_lr = 6e-4 
    min_lr = max_lr * 0.01
    warmup_steps = 2
    max_steps = 200 # Depends on the dataset + batch size... TODO
    steps_per_epoch = 20
    num_epochs = 5
    global_steps = max_epoch * max_steps

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

    run_wandb = wandb.init(
        project=os.environ["PROJECT"],
        entity=os.environ["ENTITY"],
        name=f"run_{max_epoch}",
        config={
            "dataset": dataset,
            "device": device,
            "batch_size": B,
            "sequence_length": T,
            "total_batch_size": total_batch_size,
            "max_lr": max_lr,
            "min_lr": min_lr, 
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
            "steps_per_epoch": steps_per_epoch,
            "num_epochs": num_epochs,
            "max_epoch": max_epoch,
            "model_name": "gpt_2",
            "model_config": {
                "vocab_size": gpt_config.vocab_size,
                "block_size": gpt_config.block_size,
                "n_layer": gpt_config.n_layer,
                "n_head": gpt_config.n_head,
                "n_embd": gpt_config.n_embd,
                "dropout": gpt_config.dropout,
                "bias": gpt_config.bias
            }
        }
    )

    if profile_run:
        logs_path = os.path.join(parent_path.parent.resolve(), "logs")
        if (not os.path.exists(logs_path)):
            os.makedirs(logs_path)

        prof_schedule = schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1
        )

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=prof_schedule,
            on_trace_ready=tensorboard_trace_handler(os.path.join(logs_path, "profiler_logs")),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=False
        )

        prof.start()
    # TODO: Logging, optimizer
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

    for epoch in range(num_epochs):
        t0_epoch = time.time()
        print(f"starting epoch {epoch}")
        for step in range(steps_per_epoch):
            t0 = time.time()

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

            lr = get_lr(global_steps)
            global_steps += 1

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

            if profile_run:
                prof.step()
        
        t1_epoch = time.time()
        dt_epoch = t1_epoch - t0_epoch

        if checkpoint_path is not None:
            
            if max_epoch == -1:
                max_epoch = 0
            else:
                max_epoch += 1

            path = os.path.join(parent_path.resolve(), checkpoint_path)
            os.makedirs(path, exist_ok=True)

            try:
                torch.save(model.state_dict(), os.path.join(path, f"gpt2_epoch_{max_epoch}.pth"))
            except Exception as e:
                print(f"couldn't save model for epoch {max_epoch}: {e}")


        print("starting evaluation: ")

        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 5
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

            print(f"validation loss: {val_loss_accum.item():.4f}")    

        print("generating random tokens: ")

        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:

            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen)
                
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix) 
                xgen = torch.cat((xgen, xcol), dim=1)
        
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens) 
            print(f"rank {ddp_rank} sample {i}: {decoded}")

        print(f"finishing epoch {epoch} | time: {dt_epoch:.2f}s")
                
    run_wandb.finish()

    

if __name__ == "__main__":
    train_gpt(checkpoint_path=os.path.join(parent_path.resolve(), "checkpoints"))