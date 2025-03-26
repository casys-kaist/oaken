from src.model import get_model_path
from src.util import get_model_device_map
import torch

from transformers import AutoConfig

KIVI_K_BITS = 4
KIVI_V_BITS = 4
KIVI_GROUP_SIZE = 128 # MAX SIZE IS 128
KIVI_RESIDUAL_LENGTH = 128

def get_kivi_eval_model(device, tokenizer, model, model_size, gpu_count=1, gpu_start_idx=0, do_eval=False):
    config = AutoConfig.from_pretrained(get_model_path(model, model_size),)
    config.k_bits = KIVI_K_BITS
    config.v_bits = KIVI_V_BITS
    config.group_size = KIVI_GROUP_SIZE
    config.residual_length = KIVI_RESIDUAL_LENGTH

    device_map = get_model_device_map(
        model,
        gpu_count,
        config.num_hidden_layers,
        gpu_start_idx,
    )

    match model:
        case "llama" | "llama2":
            if do_eval:
                from KIVI.models.llama_kivi_eval import LlamaForCausalLM_KIVI_eval as LlamaModelKIVI
            else:
                from KIVI.models.llama_kivi import LlamaForCausalLM_KIVI as LlamaModelKIVI
            config.use_flash = False

            return LlamaModelKIVI.from_pretrained(
                get_model_path(model, model_size),
                config=config,
                torch_dtype=torch.float16,
                device_map=device_map,
            )
        case "opt":
            if do_eval:
                from KIVI.models.opt_kivi_eval import OPTForCausalLM_KIVI_eval as OPTModelKIVI
            else:
                from KIVI.models.opt_kivi import OPTForCausalLM_KIVI as OPTModelKIVI
            config._attn_implementation = "eager"

            return OPTModelKIVI.from_pretrained(
                    get_model_path(model, model_size),
                    config=config,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                )
        case "mistral":
            if do_eval:
                from KIVI.models.mistral_kivi_eval import MistralForCausalLM_KIVI_eval as MistralModelKIVI
            else:
                from KIVI.models.mistral_kivi import MistralForCausalLM_KIVI as MistralModelKIVI
            config.use_flash = False

            return MistralModelKIVI.from_pretrained(
                get_model_path(model, model_size),
                config=config,
                torch_dtype=torch.float16,
                device_map=device_map,
            )
        case "mixtral":
            if do_eval:
                from KIVI.models.mixtral_kivi_eval import MixtralForCausalLM_KIVI_eval as MistralModelKIVI
            else:
                from KIVI.models.mixtral_kivi import MixtralForCausalLM_KIVI as MistralModelKIVI
            config._attn_implemenmtation = "eager"

            return MistralModelKIVI.from_pretrained(
                get_model_path(model, model_size),
                config=config,
                torch_dtype=torch.float16,
                device_map=device_map,
            )
        case _:
            raise ValueError(f"Model {model} not supported.")

def get_kivi_model(device, tokenizer, model, model_size, gpu_count=1, gpu_start_idx=0):
    pass

def kivi_main(args, model, tokenizer, device, runner):
    runner(args, model, tokenizer, device)