import torch
import argparse
import json
from functools import partial
from tqdm import tqdm
from datasets import load_dataset

from src.model import get_model
from src.tokenizer import get_tokenizer

def main(args):
    print("= Run Configuration ===================")
    print(f"Model name: {args.model}")
    print(f"Model size: {args.model_size}")
    print(f"Task: {args.task}")
    print(f"Single run: {args.single_run}")
    print(f"Quantization method: {args.quant_method}")
    print(f"Quantizer path: {args.quantizer_path}")
    print(f"Used GPUs: {args.gpu_start_idx} - {args.gpu_start_idx + args.gpu_count - 1}")
    print("=======================================")

    torch_device = "cuda"
    tokenizer = get_tokenizer(args.model, args.model_size)

    if args.quant_method == "kivi":
        from kivi_main import get_kivi_eval_model
        model = get_kivi_eval_model(torch_device, tokenizer, args.model, args.model_size, args.gpu_count, args.gpu_start_idx)
    elif args.quant_method == "tender":
        from tender_main import get_tender_model
        model = get_tender_model(torch_device, tokenizer, args.model, args.model_size, args.gpu_count, args.gpu_start_idx)
    else:
        model = get_model(torch_device, tokenizer, args.model, args.model_size, args.gpu_count, args.gpu_start_idx)

    match args.quant_method:
        case "oaken":
            from oaken_main import multi_group_oaken_main
            print("Running oaken")
            multi_group_oaken_main(args, model, tokenizer, torch_device, eval_perplexity)
        case "kvquant":
            from kvquant_main import kvquant_main
            print("Running KVQuant")
            kvquant_main(args, model, tokenizer, torch_device, eval_perplexity)
        case "qserve":
            from qserve_main import qserve_main
            print("Running QServe")
            qserve_main(args, model, tokenizer, torch_device, eval_perplexity)
        case "kivi":
            from kivi_main import kivi_main
            print("Running KIVI")
            kivi_main(args, model, tokenizer, torch_device, eval_perplexity)
        case "tender":
            from tender_main import tender_main
            print("Running Tender")
            tender_main(args, model, tokenizer, torch_device, eval_perplexity)
        case _:
            print("Running Original Model")
            eval_perplexity(args, model, tokenizer, torch_device)

def eval_perplexity(args, model, tokenizer, device):
    if hasattr(args, "single_run") and args.single_run:
        input_prompt = input("Enter the input prompt: ")
        assert(input_prompt != "")
        input_tensor = tokenizer.encode(input_prompt, return_tensors="pt").to(device)
        output = model.generate(input_tensor, max_length=100)
        print(f"Model output: {tokenizer.decode(output[0], skip_special_tokens=True)}")
        return

    # Code from huggingface: https://huggingface.co/docs/transformers/ko/perplexity
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    max_length = 2048
    stride = 2048
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print(ppl)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the evaluation of a model on a set of tasks.")
    parser.add_argument("-t", "--task",
                      type=str, required=False, dest="task", help="The task to evaluate the model on.")
    parser.add_argument("--single-run",
                      action="store_true", dest="single_run", help="Whether to run the inference only once.")
    parser.add_argument("-m", "--model", default="gpt2",
                      type=str, required=False, dest="model", help="The model to evaluate.")
    parser.add_argument("-s", "--size",
                      type=str, required=False, dest="model_size", help="The size of the model to evaluate.")
    
    parser.add_argument("-q", "--quantizer",
                        type=str, required=False, dest="quantizer_path", help="channel-wise quantization information.")
    parser.add_argument("--quant-method",
                      type=str, required=True, dest="quant_method", help="Output file path for activation stats.")
    
    # Arguments for Oaken
    parser.add_argument("-f", "--outlier_frac", default=0.01,
                        type=float, required=False, dest="outlier_frac", help="activation outlier percentage.")
    parser.add_argument("--quant-outlier",
                        action="store_true", dest="quant_outlier", help="Whehter to quantize outliers.")
    
    # Arguments for GPU configuration
    parser.add_argument("--gpu-start-idx", default=0,
                        type=int, dest="gpu_start_idx", help="The index of the first GPU to use.")
    parser.add_argument("--gpu-count", default=1,
                        type=int, dest="gpu_count", help="The number of GPUs to use.")
    
    args = parser.parse_args()

    main(args)
