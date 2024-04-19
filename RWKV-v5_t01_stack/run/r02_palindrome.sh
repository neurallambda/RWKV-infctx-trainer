#!/bin/bash

##################################################
#
#  Sample training pipeline
#
#  USAGE:
#    # 1.) Adjust env vars in this file, eg file paths
#    # 2.) Adjust config.yaml paths, eg: ./run/r02/checkpoint
#    cd RWKV-v5
#    chmod +x run/r02_palindrome.sh
#    ./run/r02_palindrome.sh
#
#  OUTPUT:
#    run/r02/checkpoint/  # saved and checkpointed models
#    run/r02/datapath/    # cached prepared dataset
#
##################################################

# Project prefix, for wandb and filename logging
# follow the format of "discordhandle"-"shortprojectname"
export PROJECT_PREFIX="r02_palindrome"
export ROOT_DIR="."
export PROJECT_DIR="${ROOT_DIR}/run/r02"
export HF_DATASETS_OFFLINE="1"

# RWKV_NO_CUDA=0 (poor naming) to use CUDA in infctx
# or RWKV_TORCH_COMPILE=1 to speed up the non-cuda to near CUDA speed
export RWKV_NO_CUDA=0

# Model version you are using, use v5 or v4 respectively
export MODEL_VERSION="v5"

# Deepspeed strategy to use, you can leave this unchanged
export DEEPSPEED_STRAT="deepspeed_stage_1"
export GPU_DEVICES="auto"
export ENABLE_WANDB=False

# Prefixes we will be using
export WANDB_PREFIX="${PROJECT_PREFIX}"

if [ "$ENABLE_WANDB" = "True" ]; then
    export WANDB_MODE="online"
else
    export WANDB_MODE="disabled"
fi

echo "DEEPSPEED_STRAT: ${DEEPSPEED_STRAT}"
echo "ENABLE_WANDB: ${ENABLE_WANDB}"
echo "GPU_DEVICES: ${GPU_DEVICES}"
echo "PROJECT_DIR: ${PROJECT_DIR}"

# Configure the initial model name you are using
export INIT_MODEL_NAME="init.pth"

# Setup the required project directories
mkdir -p "${PROJECT_DIR}/datapath/"
mkdir -p "${PROJECT_DIR}/checkpoint/"

##################################################
# PARAMS

export S_STACK_IX="1"
export S_NOISE="0.3"


# echo "##################################################"
# echo "INITIALIZING"
# python "${ROOT_DIR}/init_model.py" \
#     --n_layer 4 --n_embd 256 \
#     --vocab_size world --skip-if-exists \
#     "${PROJECT_DIR}/checkpoint/${INIT_MODEL_NAME}"


# # echo "##################################################"
# # echo "PRELOADING DATASET"
# # # python "preload_datapath.py" "run/r02/config.yaml"
# # python "${ROOT_DIR}/preload_datapath.py" "${PROJECT_DIR}/config.yaml"


# echo "##################################################"
# echo "TRAINING"

# python "${ROOT_DIR}/lightning_trainer.py" fit \
#     -c "${PROJECT_DIR}/config.yaml" \
#     --trainer.logger.init_args.name="${WANDB_PREFIX} training (${DEEPSPEED_STRAT})" \
#     --trainer.strategy="${DEEPSPEED_STRAT}" \
#     --trainer.devices="${GPU_DEVICES}" \
#     --trainer.callbacks.init_args.dirpath="${PROJECT_DIR}/checkpoint" \
#     --model.load_model="${PROJECT_DIR}/checkpoint/${INIT_MODEL_NAME}"


echo "##################################################"
echo "EXPORTING"
python "${ROOT_DIR}/export_checkpoint.py" "${PROJECT_DIR}/checkpoint/last.ckpt" "${PROJECT_DIR}/checkpoint/final.pth"
ls -alh "${PROJECT_DIR}/checkpoint/final.pth"


echo "##################################################"
echo "TESTING"
python "${ROOT_DIR}/dragon_test.py" "${PROJECT_DIR}/checkpoint/final.pth" "cuda fp32"
