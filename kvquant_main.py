import pickle

from kvquant.quant.kvquant.modelutils import *
from kvquant.quant.kvquant.datautils import *
from kvquant.quant.kvquant.simquant_module_quantizer import *

def kvquant_main(args, model, tokenizer, torch_device, runner):
    with open(args.quantizer_path, 'rb') as handle:
        quantizers = pickle.load(handle)

    perchannelquant = {}
    pertokenquant = {}

    perchannel_match = ["k_proj"]
    pertoken_match = ["v_proj"]

    print(quantizers.keys())

    for k in quantizers.keys():
        for p in perchannel_match:
            if p in k:
                perchannelquant[k] = quantizers[k]
        for p in pertoken_match:
            if p in k:
                pertokenquant[k] = quantizers[k]
    
    #per-vector quant - Key
    make_quant_sim(
        model,
        perchannelquant,
        4,
        perchannel=True,
        include_sparse=True,
        sparsity_threshold=(1-args.outlier_frac),
        dynamicquantization=False,
        nuq=False,
        nf_nuq=True,
        norm=False,
        cap_outliers=-1,
        first_few_fp16=-1,
        clamp=False
    )

    #per-vector quant - Value
    make_quant_sim(
        model,
        pertokenquant,
        4,
        perchannel=False,
        include_sparse=True,
        sparsity_threshold=(1-args.outlier_frac),
        dynamicquantization=True,
        nuq=False,
        nf_nuq=True,
        norm=False,
        cap_outliers=-1,
        first_few_fp16=-1,
        clamp=False
    )
    return runner(args, model, tokenizer, torch_device)
