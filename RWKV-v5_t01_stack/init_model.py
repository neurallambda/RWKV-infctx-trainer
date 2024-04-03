'''

----------
EXAMPLE PARAMS:

emb.weight

blocks.0.ln1.weight
blocks.0.ln1.bias
blocks.0.ln2.weight
blocks.0.ln2.bias
blocks.0.ln0.weight
blocks.0.ln0.bias
blocks.0.att.time_mix_k
blocks.0.att.time_mix_v
blocks.0.att.time_mix_r
blocks.0.att.time_mix_g
blocks.0.att.time_decay
blocks.0.att.time_faaaa
blocks.0.att.receptance.weight
blocks.0.att.key.weight
blocks.0.att.value.weight
blocks.0.att.output.weight
blocks.0.att.gate.weight
blocks.0.att.ln_x.weight
blocks.0.att.ln_x.bias
blocks.0.ffn.time_mix_k
blocks.0.ffn.time_mix_r
blocks.0.ffn.key.weight
blocks.0.ffn.receptance.weight
blocks.0.ffn.value.weight

ln_out.weight
ln_out.bias
head.weight

stack.sharp
stack.push.0.weight
stack.push.0.bias
stack.push.2.weight
stack.pop.0.weight
stack.pop.0.bias
stack.pop.2.weight
stack.nop.0.weight
stack.nop.0.bias
stack.nop.2.weight


'''

import argparse, math, os
import torch.nn as nn
import torch
from src.model import RWKV

def init_model(
        layers, embedding_size, vocab_size, output_model_path,
        skip_if_exists=False, safe_init=False, emb_scale=0.0001
        # existing_model_path=None
        ):
    # Insert your own function behavior here
    print(f"---- Initializing model ----")
    print(f'No of layers: {layers}')
    print(f'Embedding size: {embedding_size}')
    print(f'Output model path: {output_model_path}')
    print(f'Vocab size: {vocab_size}')
    print(f'Emb scale: {emb_scale}')
    # print(f'Existing model path: {existing_model_path}')
    print(f'Note: this process takes a significant time (and ram) for large models')
    print(f"---- ----- ----")

    # Check if the model exists
    if skip_if_exists and os.path.exists(output_model_path):
        print(f"Model exists, skipping init_model")
        return

    # Enforce safe_init if skip_if_exists is set
    if skip_if_exists:
        safe_init = True

    # Ensure the parent dir exists
    parent_dir = os.path.dirname(output_model_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # Setup the RWKV model, with the special init_model str
    # this disable the loading of the init model file
    model = RWKV(n_layer=layers,
                 n_embd=embedding_size, vocab_size=vocab_size,
                 load_model=".//<#|=@%!$init_model$!%@=|#>//.",
                 ctx_len=1)

    # Modified init code, from the original init code
    m = {}
    for n in model.state_dict():

        # Iterate each parameter group in state_dict
        p = model.state_dict()[n]
        shape = p.shape
        gain = 1.0
        scale = 1.0

        if ("ln_" in n or
            ".ln" in n or
            "time_" in n or
            "_mask" in n or
            "pos_emb" in n or
            '.mask.' in n or
            'stack' in n):
            if 'ln_x.weight' in n:
                # Special ln_x init
                layer_scale = (1+int(n.split('.')[1])) / layers
                m[n] = (p * 0.0) + (layer_scale ** 0.7)
            else:
                # Skip custom init for these layers
                m[n] = p
            # print(f"{str(m[n].shape).ljust(5)} {n}")
        else:
            if n == "emb.weight":
                # scale = -1 * self.args.lr_init
                scale = -1 * abs(emb_scale)
            else:
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                for kk in [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']:
                    if kk in n:
                        scale = 0
                if n == "head.weight":
                    scale = 0.5
                if "head_k." in n:
                    scale = 0.1
                if "head_q." in n:
                    scale = 0

            print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {n}")

            # Reinitialize as empty params
            m[n] = torch.empty((shape[0], shape[1]))
            # With the specified vlaue ranges
            if scale == 0:
                nn.init.zeros_(m[n])
            elif scale < 0:
                nn.init.uniform_(m[n], a=scale, b=-scale)
            else:
                nn.init.orthogonal_(m[n], gain=gain * scale)

        # Ensure its mapped as a CPU & BF16
        m[n] = m[n].cpu()
        m[n] = m[n].bfloat16()

    # Save the model
    if safe_init:
        # Save as tmp file, then move to the output path
        torch.save(m, output_model_path+".tmp")
        os.rename(output_model_path+".tmp", output_model_path)
    else:
        # Save directly
        torch.save(m, output_model_path)

def main():
    parser = argparse.ArgumentParser(description='CLI tool for model handling')
    parser.add_argument('--n_layer', type=int, help='Number of layers')
    parser.add_argument('--n_embd',  type=int, help='Embedding size')
    parser.add_argument('--vocab_size', type=str, help="Vocab size for the model as an int, alternativey use 'neox' or 'world' if using their respective tokenizer", default="neox")
    parser.add_argument('--skip-if-exists', type=bool, action=argparse.BooleanOptionalAction, default=False, help='Skip the init if the model already exists, enables --safe-init if set')
    parser.add_argument('--safe-init', type=bool, action=argparse.BooleanOptionalAction, default=False, help='Init in safe mode, where the model is first init as a tmp file, before overwritting/moving to the output path')
    parser.add_argument('--emb-scale', type=float, default=0.0001, help='Embedding weight scale, default is 0.0001')

    # (todo) implement in the future, to support model resizing
    # parser.add_argument('--existing_model_path', type=str, help='Existing model path', default=None)
    parser.add_argument('output_model_path', type=str, help='Output model file path')

    # Parse the args
    args = parser.parse_args()

    # Parse the vocab_size
    vocab_size = args.vocab_size
    if vocab_size == "neox":
        vocab_size = 50277
    elif vocab_size == "world":
        vocab_size = 65536
    else:
        vocab_size = int(vocab_size)

    init_model(
        args.n_layer, args.n_embd, vocab_size, args.output_model_path,
        skip_if_exists=args.skip_if_exists, safe_init=args.safe_init,
        emb_scale=args.emb_scale
    ) #, args.existing_model_path

if __name__ == "__main__":
    main()
