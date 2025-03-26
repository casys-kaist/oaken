import torch
import argparse

from src.model import get_model
from src.tokenizer import get_tokenizer
from src.evaluate import evaluate_task

def main(args):
    print("= Run Configuration ===================")
    print(f"Model name: {args.model}")
    print(f"Model size: {args.model_size}")
    print(f"Task: {args.task}")
    print(f"Batch size: {args.batch_size}")
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
            print("Running multigroup_oaken")
            multi_group_oaken_main(args, model, tokenizer, torch_device, run_model)
        case "kvquant":
            from kvquant_main import kvquant_main
            print("Running KVQuant")
            kvquant_main(args, model, tokenizer, torch_device, run_model)
        case "qserve":
            from qserve_main import qserve_main
            print("Running QServe")
            qserve_main(args, model, tokenizer, torch_device, run_model)
        case "kivi":
            from kivi_main import kivi_main
            print("Running KIVI")
            kivi_main(args, model, tokenizer, torch_device, run_model)
        case "tender":
            from tender_main import tender_main
            print("Running Tender")
            tender_main(args, model, tokenizer, torch_device, run_model)
        case _:
            print("Running without quantization")
            run_model(args, model, tokenizer, torch_device)
    
def run_model(args, model, tokenizer, device):
    # if args.single_run:
    #     input_prompt = input("Enter the input prompt: ")
    #     assert(input_prompt != "")
    #     input_tensor = tokenizer.encode(input_prompt, return_tensors="pt").to(device)
    #     output = model.generate(input_tensor, max_length=200)
    #     print(f"Model output: {tokenizer.decode(output[0], skip_special_tokens=True)}")
    # else:
    result = evaluate_task(model, tokenizer, args.batch_size, [args.task,])
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the evaluation of a model on a set of tasks.")
    parser.add_argument("-t", "--task",
                      type=str, required=False, dest="task", help="The task to evaluate the model on.")
    parser.add_argument("-m", "--model", default="gpt2",
                      type=str, required=False, dest="model", help="The model to evaluate.")
    parser.add_argument("-s", "--size",
                      type=str, required=False, dest="model_size", help="The size of the model to evaluate.")
    parser.add_argument("-b", "--batch-size", default=1,
                      type=int, required=False, dest="batch_size", help="The batch size to use for the evaluation.")
    parser.add_argument("--quant-method", default="",
                      type=str, required=True, dest="quant_method", help="Output file path for activation stats.")

    # Used commonly for all quantization methods
    parser.add_argument("-q", "--quantizer",
                        type=str, required=False, dest="quantizer_path", help="channel-wise quantization information.")
    
    # Used only for oaken and kvquant
    parser.add_argument("-f", "--outlier_frac", default=0.01,
                        type=float, required=False, dest="outlier_frac", help="activation outlier percentage.")
    
    # Used only for oaken
    parser.add_argument("--quant-outlier", default=True,
                        action="store_true", dest="quant_outlier", help="Whehter to quantize outliers.")
    parser.add_argument("--single-run",
                      action="store_true", dest="single_run", help="Whether to run the inference only once.")
    
    # Arguments for GPU configuration
    parser.add_argument("--gpu-start-idx", default=0,
                        type=int, dest="gpu_start_idx", help="The index of the first GPU to use.")
    parser.add_argument("--gpu-count", default=1,
                        type=int, dest="gpu_count", help="The number of GPUs to use.")
    
    args = parser.parse_args()

    main(args)
