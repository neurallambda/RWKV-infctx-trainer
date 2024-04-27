# Notes on RWKV v5 Stack 01 Experiments


ERROR

ImportError: /home/josh/_/neurallambda/.env/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so: undefined symbol: ncclCommRegister

Build nccl yourself, then:

LD_LIBRARY_PATH=../tools/nccl/build/lib:$LD_LIBRARY_PATH python
