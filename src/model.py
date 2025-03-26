import torch
from src.util import get_model_device_map
from transformers import AutoConfig

MODEL_STORAGE_PREFIX = "/data/models"
MODEL_STORATE_POSTFIX = ""

def get_model_path(model: str, size: str):
    match model:
        case "llama" | "llama2":
            return f"{MODEL_STORAGE_PREFIX}/llama2-{size}/{MODEL_STORATE_POSTFIX}"
        case "opt":
            return f"{MODEL_STORAGE_PREFIX}/opt-{size}/{MODEL_STORATE_POSTFIX}"
        case "mistral":
            return f"{MODEL_STORAGE_PREFIX}/mistral-{size}/{MODEL_STORATE_POSTFIX}"
        case "mixtral":
            return f"{MODEL_STORAGE_PREFIX}/mixtral-{size}/{MODEL_STORATE_POSTFIX}"
        case _:
            raise ValueError(f"Model {model} not supported.")

def get_model(device, tokenizer, model: str, size: str, gpu_count=1, gpu_start_idx=0):
    config = AutoConfig.from_pretrained(get_model_path(model, size))

    device_map = get_model_device_map(
        model,
        gpu_count,
        config.num_hidden_layers,
        gpu_start_idx,
    )

    match model:
        case "gpt2":
            from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
            return GPT2LMHeadModel.from_pretrained("gpt2",
                                                torch_dtype=torch.float16,
                                                pad_token_id=tokenizer.eos_token_id,
                                                use_cache=True,
                                                device_map = device_map,
                                            )
        case "llama" | "llama2":
            from transformers.models.llama.modeling_llama import LlamaForCausalLM
            return LlamaForCausalLM.from_pretrained(get_model_path(model, size),
                                                torch_dtype=torch.float16,
                                                use_cache=True,
                                                pad_token_id=tokenizer.eos_token_id,
                                                device_map=device_map,
                                            )
        case "opt":
            from transformers.models.opt.modeling_opt import OPTForCausalLM
            return OPTForCausalLM.from_pretrained(get_model_path(model, size),
                                                torch_dtype=torch.float16,
                                                use_cache=True,
                                                pad_token_id=tokenizer.eos_token_id,
                                                device_map=device_map,
                                            )
        case "mistral":
            from transformers.models.mistral.modeling_mistral import MistralForCausalLM
            return MistralForCausalLM.from_pretrained(get_model_path(model, size),
                                                torch_dtype=torch.float16,
                                                use_cache=True,
                                                pad_token_id=tokenizer.eos_token_id,
                                                device_map=device_map,
                                            )
        case "mixtral":
            from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM
            return MixtralForCausalLM.from_pretrained(get_model_path(model, size),
                                                torch_dtype=torch.float16,
                                                use_cache=True,
                                                pad_token_id=tokenizer.eos_token_id,
                                                device_map=device_map,
                                            )
        case _:
            raise ValueError(f"Model {model} not supported.")
