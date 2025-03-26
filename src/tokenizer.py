def get_tokenizer(model: str, size: str):
    if model == "gpt2":
        return get_gpt2_tokenizer()
    elif model in ["opt", "opt_our_quant"]:
        return get_opt_tokenizer(size)
    elif model in ["llama", "llama2"]:
        return get_llama_tokenizer(size)
    elif model == "mistral":
        return get_mistral_tokenizer(size)
    elif model == "mixtral":
        return get_mixtral_tokenizer(size)
    else:
        raise ValueError(f"Model {model} not supported.")

def get_gpt2_tokenizer():
    from transformers import GPT2Tokenizer
    return GPT2Tokenizer.from_pretrained("gpt2")

def get_opt_tokenizer(size: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(f"facebook/opt-{size}", use_fast=False)

def get_llama_tokenizer(size: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(f"meta-llama/llama-2-{size}-hf", use_fast=False)

def get_mistral_tokenizer(size: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(f"mistralai/Mistral-7B-v0.3", use_fast=False)

def get_mixtral_tokenizer(size: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(f"mistralai/Mixtral-8x7B-v0.1", use_fast=False)