# Oaken: Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization

Oaken is an accleration solution that achieves high accuracy and high performance simultaneously through co-designing algorithm and hardware, leveraging online-offline hybrid KV cache quantization algorithm and dedicated quantization/dequantization hardware modules.

This repository provides source code for evaluating the inference accuracy of Oaken and other baselines.
Running the evaluation code requires Python 3.10 or later and CUDA 12.1 or later.

## 1. Preparing environments

### 1.1. Build Docker image

We provide a Dockerfile to build a container with Python 3.10, CUDA 12.1, and Miniconda pre-installed.
Run the following command to build the Docker image and create the corresponding container.

```sh
$ docker build --force-rm -t oaken-ae-img .
$ docker run                              \
        --name oaken-ae-container         \
        -it                               \
        --gpus '"device=[GPU LIST]"'      \
        oaken-ae-img 
```

### 1.2. Configurating Huggingface Model Path

1. Open `src/model.py`.
2. Set `MODEL_STORAGE_PREFIX`, and `MODEL_STORATE_POSTFIX` to the directory where huggingface models are stored.
3. Huggingface model directory names should follow the format `{model_name}-{model_size}`, such as llama2-7b.

### 1.3. Download Huggingface Model

1. Put your Huggingface access token to the variable `HF_TOKEN` in `model_downloader.py`. (Some of the models might require indiviual access grant) 
2. Select models to download by setting the variables `DOWNLOAD_*` in `model_downloader.py`.
3. Run the following command to download LLMs from Huggingface.

```sh
$ pip install huggingface_hub
$ python3 model_downloader.py
``` 

## 2. Oaken, KVQuant, QServe, Tender

### 2.1.1. Install Dependencies
Oaken, KVQuant, QServe, and Tender share the same installation commands.

```sh
$ git submodule update --init --recursive

$ conda create -n oaken python=3.10
$ conda activate oaken
$ pip install torch protobuf sentencepiece

$ pushd transformers
$ pip install -e .
$ popd

$ pushd lm-evaluation-harness
$ pip install -e .
$ popd

$ pushd kvquant/quant
$ pip install -e .
$ popd
$ pip install flash-attn --no-build-isolation
```

### 2.2. All-in-one Scripts
You can run the entire accuracy evaluation to get the results for Table 2 with the following command.
Please configure for list of models and number of GPUs that will be used for evaluation.
**Note that running the entire accuracy evaluation with this script takes very long time.**

```sh
$ python3 scripts/accuracy_oaken.py
$ python3 scripts/accuracy_kvquant.py
$ python3 scripts/accuracy_qserve.py
$ python3 scripts/accuracy_tender.py
```

Use the following command to get the results for Figure 12(a).

```sh
$ python3 explore_oaken.py
```

The following sections describe the instructions for running individual models and benchmarks.

### 2.3.1. Preparing Oaken
You can run offline profiling for Oaken with the following command.

```sh
$ python3 oaken_preprocess_activation.py \
    -m [MODEL NAME]     \
    -s [MODEL SIZE]     \
    -t wikitext         \
    -f 0.04 0.9 0.06    \
    -o quantizer/oaken/[OUTPUT QUANTIZER].json 
```

Example command:
```sh
$ python3 oaken_preprocess_activation.py \
    -m llama2           \
    -s 7b               \
    -t wikitext         \
    -f 0.04 0.9 0.06    \
    -o quantizer/oaken/llama2-7b.json
```

### 2.3.2 Preparing KVQuant

Run profiling process for KVQuant with the following command.
Set `[MODEL PATH]` to the directory where you downloaded huggingface model.

```sh
$ cd kvquant/quant
$ CUDA_VISIBLLE_DEVICES=0 python3 llama_simquant.py \
    [MODEL PATH]                \
    --abits 4                   \
    --nsamples 16               \
    --seqlen 2048               \
    --nuq                       \
    --quantize                  \
    --include_sparse            \
    --sparsity-threshold 0.99   \
    --quantizer-path quantizer/kvquant/[OUTPUT QUANTIZER].pickle
```

Please refer to detailed instruction at https://github.com/SqueezeAILab/KVQuant/tree/main/quant.

### 2.3.3. Preparing QServe
Run profiling process for QServe with the following command.

```sh
$ python3 qserve_preprocess_activation.py \
    -m [MODEL NAME] \
    -s [MODEL SIZE] \
    -t wikitext \
    -o quantizer/qserve/[OUTPUT QUANTIZER].json 
```

### 2.3.4. Preparing Tender

You should download `val.jsonl.zst` file with the following command.

```sh
wget https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst
```

You can run preprocessing for Tender with the following command.

```sh
$ python3 tender_preprocess_activation.py \
    -m [MODEL NAME] \
    -s [MODEL SIZE] \
    -d val.jsonl.zst \
    -o quantizer/tender/[OUTPUT QUANTIZER].pt
```

### 2.4.1 Accuracy Evaluation (Perplexity)
Use `--quant-method` option to select quantization method.
You can use `--gpu-start-idx` and `--gpu-count` option to choose which GPUs will be used for the evaluation.
Specify the path for the quantizer files generated by the above commands, if required by the quantization algorithm.

```sh
$ python3 eval_perplexity.py \
    -m [MODEL NAME] \
    -s [MODEL SIZE] \
    -t wikitext \
    -q quantizer/[QUANTIZER PATH] \
    --quant-method [oaken | qserve | kvquant | tender] \
    --gpu-start-idx 0 \
    --gpu-count 1
```

### 2.4.2 Accuracy Evaluation (Zero-shot Accuracy)
Use `--quant-method` option to select quantization method.
Use `-t` option to select evaluation dataset.
You can use `--gpu-start-idx` and `--gpu-count` option to choose which GPUs will be used for the evaluation.

```sh
$ python3 eval_workload.py \
    -m [MODEL NAME] \
    -s [MODEL SIZE] \
    -t [piqa | winogrande | hellaswag] \
    -q quantizer/[QUANTIZER PATH] \
    --quant-method [oaken | qserve | kvquant | tender] \
    --gpu-start-idx 0 \
    --gpu-count 1
```

## 3. KIVI

### 3.1. Install Dependencies
```sh
$ conda create -n kivi python=3.10
$ conda activate kivi

$ cd KIVI
$ pip install -e .
$ cd quant
$ pip install -e .

$ cd ../..
$ pip install datasets

$ pushd transformers
$ pip install -e .
$ popd

$ pushd lm-evaluation-harness
$ pip install -e .
$ popd
```

### 3.2. All-in-one Script
You can run the entire accuracy evaluation to get the results for Table 2 with the following command.
Please configure for list of models and number of GPUs that will be used for evaluation.
**Note that running the entire accuracy evaluation with this script takes very long time.**

```sh
$ python3 scripts/accuracy_kivi.py
```

### 3.3.1. Accuracy Evaluation (Perplexity)

```sh
$ python3 eval_perplexity.py \
    -m [MODEL NAME] \
    -s [MODEL SIZE] \
    -t wikitext \
    --quant-method kivi \
    --gpu-start-idx 0 \
    --gpu-count 1
```

### 3.3.2. Accuracy Evaluation (Zero-shot Accuracy)

```sh
$ python3 eval_workload.py \
    -m [MODEL NAME] \
    -s [MODEL SIZE] \
    -t [piqa | winogrande | hellaswag] \
    --quant-method kivi \
    --gpu-start-idx 0 \
    --gpu-count 1
```

## 4. Atom

### 4.1. Install Dependencies

```sh
conda create -n atom python=3.10
conda activate atom
cd Atom/model
pip install -r requirements.txt
```

### 4.2.1. Accuracy Evaluation (Perplexity)

```sh
$ cd Atom
$ ./scripts/run_atom_ppl.sh [MODEL NAME] [# of GPUs]
```

Example command: 
```sh
$ ./scripts/run_atom_ppl.sh llama2-7b 1
$ ./scripts/run_atom_ppl.sh llama2-70b 4
```

### 4.2.2. Accuracy Evaluation (Zero-shot Accuracy)

```sh
$ cd Atom
$ ./scripts/run_atom_zeroshot_acc.sh [MODEL NAME] [# of GPUs]
```

## 5. Troubleshooting

### 5.1. If you encounter: `nvcc was not found`

Configure the environment variables with the following commands.

```sh
export PATH="/usr/local/cuda-[CUDA_VERSION]/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-[CUDA_VERSION]/lib64:${LD_LIBRARY_PATH}"
```

### 5.2. If you encounter: `RuntimeError: quantile() input tensor is too large`

Use smaller sampling rate rather than 1.0 by giving option `--sample-rate` or modifying `SAMPLING_RATE` variable in the script.

### 5.3. Failing KVQuant profiling with the error: `Expected all tensors to be on the same device ...`

Modify `freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)` at `src/transformers/models/llama/modeling_llama.py` to
`freqs = (inv_freq_expanded.float().to(position_ids_expanded.device) @ position_ids_expanded.float()).transpose(1, 2)`.

### 5.4. If you encounter: `cannot import name 'EncoderDecoderCache' from 'transformers'`

Install the package with the following command.

```sh
$ pip install peft==0.10.0
```