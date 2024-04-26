#!/usr/bin/env python3
import sys
import os

# ----
# This script is used to preload the huggingface dataset
# that is configured in the config.yaml file
# ----

# Check for argument, else throw error
if len(sys.argv) < 2:
    print("No arguments supplied")
    print("Usage: python3 dragon_test.py <model-path> [device] [length]") # [tokenizer]")
    sys.exit(1)

# download models: https://huggingface.co/BlinkDL
MODEL_PATH=sys.argv[1]

# If model device is not specified, use 'cuda' as default
RAW_DEVICE = "cpu fp32"
DEVICE = "cuda"
DTYPE  = "bf16"

# Get the raw device settings (if set)
if len(sys.argv) >= 3:
    RAW_DEVICE = sys.argv[2]

# Get the output length
LENGTH=200
if len(sys.argv) >= 4:
    LENGTH=int(sys.argv[3])

# Backward support for older format, we extract only cuda/cpu if its contained in the string
if RAW_DEVICE.find('cuda') != -1:
    RAW_DEVICE = 'cuda'

# The DTYPE setting
if RAW_DEVICE.find('fp16') != -1:
    DTYPE = "fp16"
elif RAW_DEVICE.find('bf16') != -1:
    DTYPE = "bf16"
elif RAW_DEVICE.find('fp32') != -1:
    DTYPE = "fp32"

# Disable torch compile for dragon test
os.environ["RWKV_TORCH_COMPILE"] = "0"

# Setup the model
from src.model import SimpleRWKV
model = SimpleRWKV(MODEL_PATH, device=DEVICE, dtype=DTYPE)

# Dummy forward, used to trigger any warning / optimizations / etc
model.completion("\nIn a shocking finding", max_tokens=1, temperature=1.0, top_p=0.7)

# Go
prompts = ["""
# Is Palindrome?:
^ a b c | c b a $
# Answer:
""",
"""
# Is Palindrome?:
^ a b c | b c a $
# Answer:
""",
"""
# Is Palindrome?:
^ f g h | h g f $
# Answer:
""",
"""
# Is Palindrome?:
^ h f h | h f h $
# Answer:
"""
]

for prompt in prompts:
    print('----------')
    model.completion(prompt.strip(), stream_to_stdout=True, max_tokens=LENGTH, temperature=0.0)

# Empty new line, to make the CLI formatting better
print("")
