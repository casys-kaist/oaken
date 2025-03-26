import torch
import argparse
import json
from functools import partial

from tender_main import get_tender_calibration_model
from src.tokenizer import get_tokenizer

@torch.no_grad()
def common_main(args):
    print("= Run Configuration ===================")
    print(f"Model name: {args.model}")
    print(f"Model size: {args.model_size}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output: {args.output_file}")
    print("=======================================")

    tokenizer = get_tokenizer(args.model, args.model_size)
    model, calibrator = get_tender_calibration_model("cpu", tokenizer, args.model, args.model_size)

    decoder_iterator = model.model.decoder.layers if args.model == "opt" else model.model.layers

    for layer in decoder_iterator:
        layer.self_attn.quant_mha = True

        layer.self_attn.q_bits = 4
        layer.mlp.q_bits = 4
    
        layer.self_attn.decomp_factor = args.decomp_factor
        layer.decomp_factor = args.decomp_factor

        layer.self_attn.chunk_size = args.chunk_size
        layer.chunk_size = args.chunk_size

    result = calibrator(model, tokenizer, args.dataset_path, 128, 2048, True)
    torch.save(result, args.output_file)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the profiling of the model")
    parser.add_argument("-m", "--model",
                      type=str, required=True, dest="model", help="The model to evaluate.")
    parser.add_argument("-s", "--size",
                      type=str, required=True, dest="model_size", help="The size of the model to evaluate.")
    parser.add_argument("-o", "--output",
                      type=str, required=True, dest="output_file", help="Output file path for activation stats.")
    parser.add_argument("-d", "--dataset-path",
                        type=str, required=True, dest="dataset_path", help=":ocation of the calibration dataset.")
    parser.add_argument("--chunk-size", default=256,
                      type=int, required=False, dest="chunk_size")
    parser.add_argument("--decomp-factor", default=8,
                        type=int, required=False, dest="decomp_factor")
    
    args = parser.parse_args()
    common_main(args)

