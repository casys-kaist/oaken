import torch
from src.util import repeat_1d

class QServeKVQuantizer:
    QUANTIZE_BITS = 4
    
    @classmethod
    def tokenwise_uniform_quantization(cls, tensor: torch.Tensor):
        maxval = torch.max(tensor, dim=2).values
        minval = torch.min(tensor, dim=2).values
        rangeval = maxval - minval
        qx = (2 ** cls.QUANTIZE_BITS - 1) / rangeval
        offset = (minval * qx).T
        qx = qx.T
        quantized = torch.round(qx * tensor - offset)
        return (quantized + offset) / qx

    @classmethod
    def QueryScale(cls, tensor: torch.Tensor, scale: torch.Tensor, n_rep: int):
        assert scale.dim() == 1
        return tensor

    @classmethod
    def KeyScaleQuantize(cls, tensor: torch.Tensor, scale: torch.Tensor, n_rep: int):
        assert scale.dim() == 1
        inv_scale_tensor = torch.diag(1 / scale).to(tensor.device)
        scale_tensor = torch.diag(scale).to(tensor.device)
        scaled_key = tensor @ inv_scale_tensor
        print(scaled_key.shape)
        quantized_key = cls.tokenwise_uniform_quantization(scaled_key)
        scaled_key = quantized_key @ scale_tensor
        return scaled_key

    @classmethod
    def ValueQuantize(cls, tensor: torch.Tensor):
        return cls.tokenwise_uniform_quantization(tensor)