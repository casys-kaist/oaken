import torch
import argparse
from functools import partial
import json
from statistics import median
import time

from src.model import get_model
from src.tokenizer import get_tokenizer
from eval_perplexity import eval_perplexity
from eval_workload import run_model

def common_main(args):
    print("= Run Configuration ===================")
    print(f"Model name: {args.model}")
    print(f"Model size: {args.model_size}")
    print(f"Activation outlier list: {args.list_fracs}")
    print(f"Task: {args.task}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.output_file}")
    print("=======================================")

    torch_device = "cuda"
    tokenizer = get_tokenizer(args.model, args.model_size)
    model = get_model(torch_device, tokenizer, args.model, args.model_size, args.gpu_count, args.gpu_start_idx)
    return torch_device, tokenizer, model

def multi_key_value_tokenwise_main(args, runner):
    print("Called multi_key_value_tokenwise_main")
    torch_device, tokenizer, model = common_main(args)
    n_layer = len(model.get_decoder().layers)
    n_quant_group: int = len(args.list_fracs)
    assert(n_quant_group > 1)
    assert(abs(sum(args.list_fracs) - 1.0) < 1e-6)

    start_time = time.monotonic()

    stat = {
        "key": [{
            "upper_threshold": [0.0 for _ in range(n_quant_group)],
            "lower_threshold": [0.0 for _ in range(n_quant_group)],
            "counter": 0,
        } for _ in range(n_layer)],
        "value": [{
            "upper_threshold": [0.0 for _ in range(n_quant_group)],
            "lower_threshold": [0.0 for _ in range(n_quant_group)],
            "counter": 0,
        } for _ in range(n_layer)],
    }

    sample_rate = args.sample_rate

    def get_value_activation_stat_hook(i, module, input, output):
        if sample_rate != 1.0:
            sampled_indices = torch.randperm(output.size(1))[:int(sample_rate * output.size(1))]
            output_float32 = output[:,sampled_indices,:].float()
        else:
            output_float32 = output.float()
        accumulated_frac = 0.0
        for idx, frac in enumerate(args.list_fracs):
            accumulated_frac += frac
            stat["value"][i]["upper_threshold"][idx] += \
                torch.quantile(output_float32, 1 - accumulated_frac / 2.0, dim=None, interpolation="higher").half().cpu().item()
            stat["value"][i]["lower_threshold"][idx] += \
                torch.quantile(output_float32, accumulated_frac / 2.0, dim=None, interpolation="lower").half().cpu().item()
        stat["value"][i]["counter"] += 1
        return output
        
    def get_key_activation_stat_hook(i, module, input, output):
        # return output
        if sample_rate != 1.0:
            sampled_indices = torch.randperm(output.size(1))[:int(sample_rate * output.size(1))]
            output_float32 = output[:,sampled_indices,:].float()
        else:
            output_float32 = output.float()
        accumulated_frac = 0.0
        for idx, frac in enumerate(args.list_fracs):
            accumulated_frac += frac
            stat["key"][i]["upper_threshold"][idx] += \
                torch.quantile(output_float32, 1 - accumulated_frac / 2.0, dim=None, interpolation="higher").half().cpu().item()
            stat["key"][i]["lower_threshold"][idx] += \
                torch.quantile(output_float32, accumulated_frac / 2.0, dim=None, interpolation="lower").half().cpu().item()
        stat["key"][i]["counter"] += 1
        return output

    # Register forward hooks to KV results
    for i, decoder in enumerate(model.get_decoder().layers):
        decoder.self_attn.k_proj.register_forward_hook(partial(get_key_activation_stat_hook, i))
        decoder.self_attn.v_proj.register_forward_hook(partial(get_value_activation_stat_hook, i))
    
    # Run inference
    runner(args, model, tokenizer, torch_device)

    # Post-Processing
    quantizer_stat = {
        "n_quant_group": n_quant_group,
        "key": {
            "upper_threshold": [[0.0 for j in range(n_quant_group)] for i in range(n_layer)],
            "lower_threshold": [[0.0 for j in range(n_quant_group)] for i in range(n_layer)],
        },
        "value": {
            "upper_threshold": [[0.0 for j in range(n_quant_group)] for i in range(n_layer)],
            "lower_threshold": [[0.0 for j in range(n_quant_group)] for i in range(n_layer)],
        }
    }
    for i in range(n_layer):
        for j in range(n_quant_group):
            quantizer_stat["key"]["upper_threshold"][i][j] = \
                (stat["key"][i]["upper_threshold"][j] / stat["key"][i]["counter"])
            quantizer_stat["key"]["lower_threshold"][i][j] = \
                (stat["key"][i]["lower_threshold"][j] / stat["key"][i]["counter"])
            quantizer_stat["value"]["upper_threshold"][i][j] = \
                (stat["value"][i]["upper_threshold"][j] / stat["value"][i]["counter"])
            quantizer_stat["value"]["lower_threshold"][i][j] = \
                (stat["value"][i]["lower_threshold"][j] / stat["value"][i]["counter"])
            
    end_time = time.monotonic()
    print(f"Elapsed time: {end_time - start_time} seconds")

    # Write result to json file
    with open(args.output_file, mode="w") as f:
        json.dump(quantizer_stat, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the profiling of the model")
    parser.add_argument("-t", "--task",
                      type=str, required=True, dest="task", help="The task to evaluate the model on.")
    parser.add_argument("-m", "--model",
                      type=str, required=True, dest="model", help="The model to evaluate.")
    parser.add_argument("-s", "--size",
                      type=str, required=True, dest="model_size", help="The size of the model to evaluate.")
    parser.add_argument("-o", "--output",
                      type=str, required=True, dest="output_file", help="Output file path for activation stats.")
    parser.add_argument("-f", "--list_fracs", nargs='+',
                      type=float, required=True, dest="list_fracs", help="List of activation outlier percentage.")
    parser.add_argument("-b", "--batch-size", default=1,
                      type=int, required=False, dest="batch_size", help="The batch size to use for the evaluation.")
    parser.add_argument("--sample-rate", default=1.0,
                        type=float, required=False, dest="sample_rate", help="Sampling rate of torch.quantile.")
    
    # Arguments for GPU configuration
    parser.add_argument("--gpu-start-idx", default=0,
                        type=int, dest="gpu_start_idx", help="The index of the first GPU to use.")
    parser.add_argument("--gpu-count", default=1,
                        type=int, dest="gpu_count", help="The number of GPUs to use.")
    
    args = parser.parse_args()

    if args.task == "wikitext":
        multi_key_value_tokenwise_main(args, eval_perplexity)
    else:
        multi_key_value_tokenwise_main(args, run_model)