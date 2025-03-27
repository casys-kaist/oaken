import os

# You can change the following parameters to run different models and workloads
# !!! Please configure the proper GPU # for each model !!!
##################################################################################
SAMPLING_RATE = 1.0
THRESHOLD_LIST = [
    "0.10 0.80 0.10",

    "0.06 0.82 0.12",
    "0.08 0.82 0.10",
    "0.10 0.82 0.08",
    "0.12 0.82 0.06",

    "0.04 0.86 0.10",
    "0.06 0.86 0.08",
    "0.08 0.86 0.06",
    "0.10 0.86 0.04",

    "0.02 0.90 0.08",
    "0.03 0.90 0.07",
    "0.04 0.90 0.06",
    "0.05 0.90 0.05",
    "0.06 0.90 0.04",

    "0.02 0.92 0.06",
    "0.03 0.92 0.05",
    "0.04 0.92 0.04",

    "0.02 0.94 0.04",
    "0.01 0.96 0.03",
]
##################################################################################

MODEL_NAME = "llama2"
MODEL_SIZE = "7b"
N_GPU = 1
EVAL_WORKLOAD_LIST = "wikitext"

NEED_SCALING = True

for idx, threshold in enumerate(THRESHOLD_LIST):
    GPU_LIST = list(range(N_GPU))
    GPU_NUM = ",".join([str(i) for i in GPU_LIST])

    # Run profiling
    if NEED_SCALING:
        print(f"Offline profiling for {MODEL_NAME}-{MODEL_SIZE} using wikitext")
        profiling_cmd = f"" + \
                    f"CUDA_VISIBLE_DEVICES={GPU_NUM} python3 oaken_preprocess_activation.py " + \
                    f"-m {MODEL_NAME} " + \
                    f"-s {MODEL_SIZE} " + \
                    f"-o quantizer/oaken/{MODEL_NAME}-{MODEL_SIZE}-{threshold.replace(' ', '_')}.json " + \
                    f"--list_fracs {threshold} " + \
                    f"-t wikitext " + \
                    f"--sample-rate {SAMPLING_RATE} " \
                    f"--gpu-count {N_GPU} " + \
                    f"> result/original/{MODEL_NAME}-{MODEL_SIZE}-wikitext-{threshold.replace(' ', '_')}.txt"  
        # print(profiling_cmd)
        os.system(profiling_cmd)
        
    print(f"Running wikitext for {MODEL_NAME}-{MODEL_SIZE}")
    wikitext_cmd = f"" + \
                f"CUDA_VISIBLE_DEVICES={GPU_NUM} python3 eval_perplexity.py " + \
                f"-m {MODEL_NAME} " + \
                f"-s {MODEL_SIZE} " + \
                f"-q quantizer/oaken/{MODEL_NAME}-{MODEL_SIZE}-{threshold.replace(' ', '_')}.json " + \
                f"--quant-outlier " + \
                f"--quant-method oaken " + \
                f"-t wikitext " + \
                f"--gpu-count {N_GPU} " + \
                f"> result/oaken/{MODEL_NAME}-{MODEL_SIZE}-wikitext-{threshold.replace(' ', '_')}.txt"  
    # print(wikitext_cmd)
    os.system(wikitext_cmd)