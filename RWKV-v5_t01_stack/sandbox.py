'''

A place to play within an interpreter


python ... --n_layer 4 --n_embd 256 --vocab_size world

'''

import neurallambda
import sys
import os
import torch
import time
import cProfile
from src.model import SimpleRWKV

MODEL_PATH = "../run/r01/checkpoint/final.pth"
DEVICE = "cuda"
DTYPE  = "bf16"
LENGTH=200
CWD = 'RWKV-v5_t01_stack'
TEMP = 0.0
if CWD not in os.getcwd():
    os.chdir(CWD)

# Disable torch compile for dragon test
# os.environ["RWKV_TORCH_COMPILE"] = "0"
os.environ['RWKV_NO_CUDA'] = '0'

try:
    already_loaded
except:
    model = SimpleRWKV(MODEL_PATH, device=DEVICE, dtype=DTYPE)
    already_loaded = True

def run_loop(prompt):
    toks = model.encode(prompt)
    state_obj = None
    out_tokens = toks.tolist()

    for i in range(LENGTH):
        logits, state_obj = model.forward(toks, state_obj)
        tok = model.sample_logits(logits, temperature=TEMP)
        toks = [tok]
        out_tokens.append(tok)
    return out_tokens

# Profile the run_loop function

prompt = 'Once upon a time'
start_time = time.time()
# cProfile.run('run_loop()', sort='tottime')
run_loop(prompt)
out_str = model.decode(out_tokens)
end_time = time.time()
print(out_str)

total_time = end_time - start_time
total_tokens = len(out_tokens)
tokens_per_second = total_tokens / total_time
print(f"Tokens per second: {tokens_per_second:.2f}")
print(f"Total trainable parameters: {sum(p.numel() for p in model.model.parameters() if p.requires_grad)}")
