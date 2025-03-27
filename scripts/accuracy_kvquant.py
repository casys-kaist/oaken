import os

# You can change the following parameters to run different models and workloads
# !!! Please configure the proper GPU # for each model !!!
##################################################################################
MODEL_LIST = \
    ["llama2",  "llama2", "llama2", "opt",  "opt", "opt", "mistral", "mixtral"]
SIZE_LIST = \
    ["7b",      "13b",    "70b",    "6.7b", "13b", "30b", "7b",      "8x7b"   ]
N_GPUS = \
    [1,         1,        4,        1,      1,     2,      1,        4        ]
EVAL_WORKLOAD_LIST  = ("wikitext", "piqa", "winogrande", "hellaswag")
MODEL_STORAGE_PREFIX = "/data/models"
BATCH_SIZE = 4
##################################################################################

NEED_SCALING = True

for idx, (model, size, n_gpu) in enumerate(zip(MODEL_LIST, SIZE_LIST, N_GPUS)):
    GPU_LIST = list(range(n_gpu))
    GPU_NUM = ",".join([str(i) for i in GPU_LIST])

    # Run profiling
    if NEED_SCALING:
        print(f"Offline profiling for {model}-{size} using wikitext")
        profiling_cmd = f"" + \
                    f"CUDA_VISIBLE_DEVICES={GPU_NUM} python3 kvquant/quant/llama_simquant.py " + \
                    f"{MODEL_STORAGE_PREFIX}/{model}-{size} " + \
                    f"--abits 4 " + \
                    f"--nsamples 16 " + \
                    f"--seqlen 2048 " + \
                    f"--nuq " + \
                    f"--quantize " + \
                    f"--include_sparse " + \
                    f"--sparsity-threshold 0.99 " + \
                    f"--quantizer-path quantizer/kvquant/{model}-{size}.pickle"
        # print(profiling_cmd)
        os.system(profiling_cmd)
        
    for workload in EVAL_WORKLOAD_LIST:
        if (workload == "wikitext"):
            print(f"Running wikitext for {model}-{size}")
            wikitext_cmd = f"" + \
                        f"CUDA_VISIBLE_DEVICES={GPU_NUM} python3 eval_perplexity.py " + \
                        f"-m {model} " + \
                        f"-s {size} " + \
                        f"-q quantizer/kvquant/{model}-{size}.pickle " + \
                        f"--quant-method kvquant " + \
                        f"-t wikitext " + \
                        f"-f 0.01 " + \
                        f"--gpu-count {n_gpu} " + \
                        f"> result/kvquant/{model}-{size}-wikitext.txt"  
            # print(wikitext_cmd)
            os.system(wikitext_cmd)

        elif (model == "llama2"):
            print(f"Running {workload} for {model}-{size}")
            workload_cmd = f"" + \
                        f"CUDA_VISIBLE_DEVICES={GPU_NUM} python3 eval_workload.py " + \
                        f"-m {model} " + \
                        f"-s {size} " + \
                        f"-q quantizer/kvquant/{model}-{size}.pickle " + \
                        f"--quant-method kvquant " + \
                        f"-t {workload} " + \
                        f"-b {BATCH_SIZE} " + \
                        f"-f 0.01 " + \
                        f"--gpu-count {n_gpu} " + \
                        f"> result/kvquant/{model}-{size}-{workload}.txt"  
            # print(workload_cmd)
            os.system(workload_cmd)