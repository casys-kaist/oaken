from transformers import AutoTokenizer
import torch

def get_tokenizer(model):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    return tokenizer

def set_decoder_device_map(device_map: dict, prefix: str, num_device: int, num_layer: int,
                   device_start_num: int = 0):
    num_layer_per_device = num_layer // num_device
    num_remaining_layers = num_layer % num_device
    
    current_layer_idx = 0
    for device_idx in range(device_start_num, device_start_num + num_device):
        for i in range(current_layer_idx, current_layer_idx + num_layer_per_device):
            device_map[f'{prefix}.{i}'] = device_idx
        current_layer_idx += num_layer_per_device
        if num_remaining_layers > 0:
            device_map[f'{prefix}.{current_layer_idx}'] = device_idx
            current_layer_idx += 1
            num_remaining_layers -= 1
    return

def get_model_device_map(model_name: str, num_device: int, num_layer: int,
                         device_start_num: int=0):
    device_map = dict()
    LAST_GPU_IDX = device_start_num + num_device - 1
    match model_name:
        case "llama" | "llama2":
            device_map = {
                'model.embed_tokens': device_start_num,
                'model.norm': LAST_GPU_IDX,
                'lm_head': LAST_GPU_IDX,  
            }

            set_decoder_device_map(
                device_map,
                'model.layers',
                num_device,
                num_layer,
                device_start_num,
            )
        case "opt":
            device_map = {
                'model.decoder.embed_tokens': LAST_GPU_IDX,
                'model.decoder.embed_positions': LAST_GPU_IDX,
                'lm_head': LAST_GPU_IDX,
                'model.decoder.final_layer_norm': LAST_GPU_IDX,
            }

            set_decoder_device_map(
                device_map,
                'model.decoder.layers',
                num_device,
                num_layer,
                device_start_num,
            )
        case "mistral":
            device_map = {
                'model.embed_tokens': device_start_num,
                'model.norm': LAST_GPU_IDX,
                'lm_head': LAST_GPU_IDX,  
            }

            set_decoder_device_map(
                device_map,
                'model.layers',
                num_device,
                num_layer,
                device_start_num,
            )
        case "mixtral":
            device_map = {
                'model.embed_tokens': device_start_num,
                'model.norm': LAST_GPU_IDX,
                'lm_head': LAST_GPU_IDX,  
            }

            set_decoder_device_map(
                device_map,
                'model.layers',
                num_device,
                num_layer,
                device_start_num,
            )
        case _:
            raise ValueError(f"Model {model_name} not supported.")
    return device_map

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)

    Borrowed from Transformers repository
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def repeat_1d(tensor: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return tensor
    num_key_value_heads = tensor.shape[0]
    return tensor[:, None].expand(num_key_value_heads, n_rep).reshape(num_key_value_heads * n_rep)