# cell_type_representation
Contains all code used in this project for benchmarking generalizability of embedding space creation.

## How to run
In *benchmark_generalizability.py* the commands needed to run the benchmark on all datasets can be found as a comment. <br> 
Worth noting is that all models only need 1 A100 GPU to run, except model2 and model3 which requires 4 A100 GPUs to run. 
Hence it can be beneficial to start training all models except model2 and model3, and later train model2 and model3 
but on 4 GPUs.