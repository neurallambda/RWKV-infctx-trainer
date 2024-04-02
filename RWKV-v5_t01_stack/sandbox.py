'''

A place to play within an interpreter

python ... --n_layer 4 --n_embd 256 --vocab_size world

'''

import neurallambda
import sys
import os
# Disable torch compile for dragon test
os.environ["RWKV_TORCH_COMPILE"] = "0"
# Must set before importing RWKV
os.environ['RWKV_NO_CUDA'] = '0'
import torch
import time
import cProfile
CWD = 'RWKV-v5_t01_stack'
if CWD not in os.getcwd():
    os.chdir(CWD)
from src.model import SimpleRWKV

##########
# Experiment Params

torch.manual_seed(152)

os.environ['S_STACK_IX'] = str(1)
os.environ['S_NOISE'] = str(0.3)

##########


MODEL_PATH = "../run/r01/checkpoint/final.pth"
# MODEL_PATH = os.path.expanduser('~/_/models/rwkv/RWKV-5-World-0.4B-v2-20231113-ctx4096.pth')
# MODEL_PATH = os.path.expanduser('~/_/models/rwkv/RWKV-5-World-3B-v2-20231113-ctx4096.pth')
# MODEL_PATH = os.path.expanduser('~/_/models/rwkv/RWKV-x060-World-1B6-v2-20240208-ctx4096.pth')

DEVICE = "cuda"
DTYPE  = "bf16"
LENGTH = 200 # max output len
TEMP = 0.0

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
        tok = model.sample_logits(logits, temperature=TEMP, top_p=0.9)
        toks = [tok]
        out_tokens.append(tok)
    return out_tokens

# Profile the run_loop function

prompt = 'Once upon a time'
start_time = time.time()
# cProfile.run('run_loop()', sort='tottime')
out_tokens = run_loop(prompt)
out_str = model.decode(out_tokens)
end_time = time.time()
print(out_str)

total_time = end_time - start_time
total_tokens = len(out_tokens)
tokens_per_second = total_tokens / total_time
print(f"Tokens per second: {tokens_per_second:.2f}")
print(f"Total trainable parameters: {sum(p.numel() for p in model.model.parameters() if p.requires_grad)}")
