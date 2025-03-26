from huggingface_hub import snapshot_download
from src.model import MODEL_STORAGE_PREFIX

HF_TOKEN = ""

DOWNLOAD_LLAMA2_7B = False
DOWNLOAD_LLAMA2_13B = False
DOWNLOAD_LLAMA2_70B = False
DOWNLOAD_OPT_6_7B = False
DOWNLOAD_OPT_13B = False
DOWNLOAD_OPT_30B = False
DOWNLOAD_MISTRAL_7B = False
DOWNLOAD_MIXTRAL_8X7B = False

if DOWNLOAD_LLAMA2_7B:
    snapshot_download(
        repo_id="meta-llama/Llama-2-7b-hf",
        local_dir=f"{MODEL_STORAGE_PREFIX}/llama2-7b",
        ignore_patterns=[
            "*.bin"
        ],
        token=HF_TOKEN
    )

if DOWNLOAD_LLAMA2_13B:
    snapshot_download(
        repo_id="meta-llama/Llama-2-13b-hf",
        local_dir=f"{MODEL_STORAGE_PREFIX}/llama2-13b",
        ignore_patterns=[
            "*.bin"
        ],
        token=HF_TOKEN
    )

if DOWNLOAD_LLAMA2_70B:
    snapshot_download(
        repo_id="meta-llama/Llama-2-70b-hf",
        local_dir=f"{MODEL_STORAGE_PREFIX}/llama2-70b",
        ignore_patterns=[
            "*.bin"
        ],
        token=HF_TOKEN
    )

if DOWNLOAD_OPT_6_7B:
    snapshot_download(
        repo_id="facebook/opt-6.7b",
        local_dir=f"{MODEL_STORAGE_PREFIX}/opt-6.7b",
        ignore_patterns=[
            "*.msgpack",
            "*.h5"
        ],
        token=HF_TOKEN
    )

if DOWNLOAD_OPT_13B:
    snapshot_download(
        repo_id="facebook/opt-13b",
        local_dir=f"{MODEL_STORAGE_PREFIX}/opt-13b",
        ignore_patterns=[
            "*.msgpack",
            "*.h5"
        ],
        token=HF_TOKEN
    )

if DOWNLOAD_OPT_30B:
    snapshot_download(
        repo_id="facebook/opt-30b",
        local_dir=f"{MODEL_STORAGE_PREFIX}/opt-30b",
        ignore_patterns=[
            "*.msgpack",
            "*.h5"
        ],
        token=HF_TOKEN
    )

if DOWNLOAD_MISTRAL_7B:
    snapshot_download(
        repo_id="mistralai/Mistral-7B-v0.3",
        local_dir=f"{MODEL_STORAGE_PREFIX}/mistral-7b",
        ignore_patterns=[
            "*.pt"
        ],
        token=HF_TOKEN
    )

if DOWNLOAD_MIXTRAL_8X7B:
    snapshot_download(
        repo_id="mistralai/Mixtral-8x7B-v0.1",
        local_dir=f"{MODEL_STORAGE_PREFIX}/mixtral-8x7b",
        ignore_patterns=[
            "*.pt"
        ],
        token=HF_TOKEN
    )