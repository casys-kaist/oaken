import torch
from src.model import get_model_path
from src.util import get_model_device_map
from transformers import AutoConfig

TENDER_QUANT_BITS = 4 # Number of bits for quantization
TENDER_DECOMP_FACTOR = 0 # Number of groups for classification
TENDER_CHUNK_SIZE = 256 # Size of row chunk

def get_tender_calibration_model(device, tokenizer, model, model_size, gpu_count=1, gpu_start_idx=0):
    kwargs = {"torch_dtype": torch.float16, "device_map": "cpu"}
    match model:
        case "llama" | "llama2":
            from tender.models.modeling_llama_tender import LlamaForCausalLM
            from tender.calibration.llama.calibration import get_scale_factor
            return LlamaForCausalLM.from_pretrained(
                get_model_path(model, model_size),
                **kwargs,
            ), get_scale_factor
        case "opt":
            from tender.models.modeling_opt_tender import OPTForCausalLM
            from tender.calibration.opt.calibration import get_scale_factor
            return OPTForCausalLM.from_pretrained(
                get_model_path(model, model_size),
                **kwargs,
            ), get_scale_factor
        case "mistral":
            from tender.models.modeling_mistral_tender import MistralForCausalLM
            from tender.calibration.mistral.calibration import get_scale_factor
            return MistralForCausalLM.from_pretrained(
                get_model_path(model, model_size),
                **kwargs,
            ), get_scale_factor
        case "mixtral":
            from tender.models.modeling_mixtral_tender import MixtralForCausalLM
            from tender.calibration.mixtral.calibration import get_scale_factor
            return MixtralForCausalLM.from_pretrained(
                get_model_path(model, model_size),
                **kwargs,
            ), get_scale_factor
        case _:
            raise ValueError(f"Model {model} not supported by Tender")

def get_tender_model(device, tokenizer, model, model_size, gpu_count=1, gpu_start_idx=0):
    config = AutoConfig.from_pretrained(get_model_path(model, model_size),)
    
    device_map = get_model_device_map(
        model,
        gpu_count,
        config.num_hidden_layers,
        gpu_start_idx,
    )

    match model:
        case "llama" | "llama2":
            from tender.models.modeling_llama_tender_eval import LlamaForCausalLM
            return LlamaForCausalLM.from_pretrained(
                get_model_path(model, model_size),
                config=config,
                torch_dtype=torch.float16,
                device_map=device_map,
            )
        case 'opt':
            from tender.models.modeling_opt_tender_eval import OPTForCausalLM
            return OPTForCausalLM.from_pretrained(
                get_model_path(model, model_size),
                config=config,
                torch_dtype=torch.float16,
                device_map=device_map,
            )
        case 'mistral':
            from tender.models.modeling_mistral_tender_eval import MistralForCausalLM
            return MistralForCausalLM.from_pretrained(
                get_model_path(model, model_size),
                config=config,
                torch_dtype=torch.float16,
                device_map=device_map,
            )
        case _:
            raise ValueError(f"Model {model} not supported by Tender")

def tender_main(args, model, tokenizer, device, runner):
    scale_factor = torch.load(args.quantizer_path)

    match args.model:
        case "llama" | "llama2":
            for layer in model.model.layers:
                attn = layer.self_attn
                prefix = "model.layers." + str(attn.layer_idx)

                attn.quant_mha = True
                attn.q_bits = TENDER_QUANT_BITS
                attn.attn_decomp_factor = TENDER_DECOMP_FACTOR
                attn.chunk_size = TENDER_CHUNK_SIZE

                attn.k_scale = scale_factor[prefix + ".self_attn" + "k_scale"].to(attn.k_proj.weight.device)
                attn.v_scale = scale_factor[prefix + ".self_attn" + "v_scale"].to(attn.v_proj.weight.device)
                print(f"k_scale: {attn.k_scale.shape}, v_scale: {attn.v_scale.shape}")
        case "opt":
            for layer in model.model.decoder.layers:
                attn = layer.self_attn
                prefix = "model.decoder.layers." + str(attn.layer_idx)

                attn.quant_mha = True
                attn.q_bits = TENDER_QUANT_BITS
                attn.attn_decomp_factor = TENDER_DECOMP_FACTOR
                attn.chunk_size = TENDER_CHUNK_SIZE

                attn.k_scale = scale_factor[prefix + ".self_attn" + "k_scale"].to(attn.k_proj.weight.device)
                attn.v_scale = scale_factor[prefix + ".self_attn" + "v_scale"].to(attn.v_proj.weight.device)
        case "mistral":
            for layer in model.model.layers:
                attn = layer.self_attn
                prefix = "model.layers." + str(attn.layer_idx)

                attn.quant_mha = True
                attn.q_bits = TENDER_QUANT_BITS
                attn.attn_decomp_factor = TENDER_DECOMP_FACTOR
                attn.chunk_size = TENDER_CHUNK_SIZE

                attn.k_scale = scale_factor[prefix + ".self_attn" + "k_scale"].to(attn.k_proj.weight.device)
                attn.v_scale = scale_factor[prefix + ".self_attn" + "v_scale"].to(attn.v_proj.weight.device)
        case _:
            raise ValueError(f"Model {args.model} not supported by Tender")

    
    # Run inference
    runner(args, model, tokenizer, device)
    return