import torch
import argparse
import json
from functools import partial

from src.model import get_model
from src.tokenizer import get_tokenizer
from src.util import repeat_kv
from eval_perplexity import eval_perplexity
from eval_workload import run_model

def common_main(args):
    print("= Run Configuration ===================")
    print(f"Model name: {args.model}")
    print(f"Model size: {args.model_size}")
    print(f"Task: {args.task}")
    print(f"Alpha: {args.alpha}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.output_file}")
    print("=======================================")

    torch_device = "cuda"
    tokenizer = get_tokenizer(args.model, args.model_size)
    model = get_model(torch_device, tokenizer, args.model, args.model_size)
    return torch_device, tokenizer, model

def profile_main(args, runner):
    device, tokenizer, model = common_main(args)
    n_layer = len(model.get_decoder().layers)
    attn_layer = model.get_decoder().layers[0].self_attn
    if hasattr(attn_layer, "num_key_value_heads"):
        n_channel = attn_layer.num_key_value_heads * attn_layer.head_dim
        n_rep = attn_layer.num_heads // attn_layer.num_key_value_heads
        print(f"num_heads: {attn_layer.num_heads}, num_key_value_heads: {attn_layer.num_key_value_heads}")
    else:
        n_channel = attn_layer.num_heads * attn_layer.head_dim
        n_rep = 1
    print(f"n_layer: {n_layer}, n_channel: {n_channel}")
    
    stat = {
        "scale": [
            torch.full((n_channel,), -1).to(device) for _ in range(n_layer)
        ]
    }

    # Get maximum value of each Key channel
    def get_key_max_stat_hook(layer_num, module, input, output):
        max_vec = torch.max(torch.abs(output), dim=1).values.squeeze()
        stat["scale"][layer_num] = torch.max(stat["scale"][layer_num].to(max_vec.device), max_vec)
        return output
    
    for i, decoder in enumerate(model.get_decoder().layers):
        decoder.self_attn.k_proj.register_forward_hook(partial(get_key_max_stat_hook, i))

    # Run inference
    runner(args, model, tokenizer, device)

    # Post-Processing
    if hasattr(attn_layer, "rotary_emb") and n_rep == 1:
        print("Rotary embedding detected. Do post-processing for rotary embedding.")
        n_channel_div = n_channel // 2
        for i in range(n_layer):
            stat["scale"][i][:n_channel_div] = torch.max(stat["scale"][i][:n_channel_div], stat["scale"][i][n_channel_div:])
            stat["scale"][i][n_channel_div:] = stat["scale"][i][:n_channel_div]
    
    for i in range(n_layer):
        stat["scale"][i] = torch.pow(stat["scale"][i], args.alpha).cpu().numpy().tolist()

    # Write result to json file
    with open(args.output_file, mode="w") as f:
        json.dump(stat, f, indent=2)

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
    parser.add_argument("-a", "--alpha", default=0.5,
                      type=float, required=False, dest="alpha", help="Alpha scaling factor")
    parser.add_argument("-b", "--batch-size", default=1,
                      type=int, required=False, dest="batch_size", help="The batch size to use for the evaluation.")
    
    args = parser.parse_args()

    if args.task == "wikitext":
        profile_main(args, eval_perplexity)
    else:
        profile_main(args, run_model)