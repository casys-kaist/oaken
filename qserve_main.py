from functools import partial
import json
from statistics import mean

from src.qserve.quantize import *

def qserve_main(args, model, tokenizer, device, runner):
    with open(args.quantizer_path, "r") as f:
        quantizer_stat = json.load(f)

        n_layer = len(model.get_decoder().layers)
        attn_layer = model.get_decoder().layers[0].self_attn
        n_channel = attn_layer.num_heads * attn_layer.head_dim
        if hasattr(attn_layer, "num_key_value_heads"):
            n_rep = attn_layer.num_heads // attn_layer.num_key_value_heads
        else:
            n_rep = 1
        
        quantizer_scale = [
            torch.tensor(quantizer_stat["scale"][i]).to(device).to(torch.half) for i in range(n_layer)
        ]

        def qserve_query_activation_hook(layer_num, module, input, output):
            return QServeKVQuantizer.QueryScale(output, quantizer_scale[layer_num], n_rep)

        def qserve_key_activation_hook(layer_num, module, input, output):
            return QServeKVQuantizer.KeyScaleQuantize(output, quantizer_scale[layer_num], n_rep)

        def qserve_value_actvation_hook(layer_num, module, input, output):
            return QServeKVQuantizer.ValueQuantize(output)

        for i, decoder in enumerate(model.get_decoder().layers):
            decoder.self_attn.q_proj.register_forward_hook(partial(qserve_query_activation_hook, i))
            decoder.self_attn.k_proj.register_forward_hook(partial(qserve_key_activation_hook, i))
            decoder.self_attn.v_proj.register_forward_hook(partial(qserve_value_actvation_hook, i))
        
        # Run inference
        runner(args, model, tokenizer, device)
        return

