import torch
from typing import Optional

class OakenQuantizer:
    QUANTIZE_BITS = 4
    OUTLIER_BITS = 5
    FLOAT_TOLERANCE = 1e-6
    
    @classmethod
    def get_outlier_threshold(cls, input_tensor: torch.Tensor, threshold_lower: float, threshold_upper: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outlier_mask = torch.logical_or(input_tensor <= threshold_lower, threshold_upper <= input_tensor)

        outlier = input_tensor * outlier_mask
        inlier = input_tensor * ~outlier_mask

        return inlier, outlier, outlier_mask

    @classmethod
    def get_multigroup_threshold(cls, input_tensor: torch.Tensor, threshold_lowers: list[float], threshold_uppers: list[float]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        group_masks = list()
        group_tensors = list()
        prev_thr_low, prev_thr_up = None, None
        for idx, (thr_low, thr_up) in enumerate(zip(threshold_lowers, threshold_uppers)):
            if idx == len(threshold_lowers) - 1: 
                # Inner-most Group
                group_masks.append(mask := torch.logical_and(input_tensor > prev_thr_low, input_tensor < prev_thr_up))
            elif (prev_thr_low is not None) and (prev_thr_up is not None):
                group_masks.append(mask := torch.logical_or(
                    torch.logical_and(prev_thr_low < input_tensor, input_tensor <= thr_low),
                    torch.logical_and(thr_up <= input_tensor, input_tensor < prev_thr_up))
                )
            else:
                # Outer-most Group
                group_masks.append(mask := torch.logical_or(               input_tensor <= thr_low, thr_up <= input_tensor))
            prev_thr_low = thr_low
            prev_thr_up = thr_up

            group_tensors.append(input_tensor * mask)

        assert(len(threshold_lowers) == len(threshold_uppers) == len(group_tensors) == len(group_masks))
        return group_tensors, group_masks
    
    @staticmethod
    def uniform_quantization_threshold(tensor, bits: int, minval: torch.Tensor, maxval: torch.Tensor):
        rangeval = maxval - minval
        qx = (2 ** bits - 1) / rangeval
        offset = minval * qx
        quantized = torch.round(qx * tensor - offset)
        quantized = torch.nan_to_num(quantized, nan=2 ** bits - 1)
        return (quantized + offset) / qx

    @staticmethod
    def uniform_quantization(tensor, bits: int):
        maxval = torch.max(tensor).cpu().item()
        minval = torch.min(tensor).cpu().item()
        return OakenQuantizer.uniform_quantization_threshold(tensor, bits, minval, maxval)
    
    @classmethod
    def downsample_mantissa(cls, tensor):
        int16_tensor = tensor.view(torch.int16)
        truncated = int16_tensor & 0b1_11111_1110_0000_00
        return truncated.view(torch.float16)

class MultiThresholdTokenwiseQuantizer(OakenQuantizer):
    @classmethod
    def downsample(cls, input_tensor: torch.Tensor, threshold_lowers: list[float], threshold_uppers: list[float],
                    quantize_outlier: bool=False, use_group_shift: bool=True) -> tuple[torch.Tensor, list[float], torch.Tensor]:
        grouped_tensors, masks = cls.get_multigroup_threshold(input_tensor, threshold_lowers, threshold_uppers)
        result_tensor = torch.zeros_like(input_tensor).to(input_tensor.device).half()

        if quantize_outlier:
            # Quantize inner-most group
            minval_tensor = torch.min(grouped_tensors[-1], dim=-1).values.unsqueeze(-1)
            maxval_tensor = torch.max(grouped_tensors[-1], dim=-1).values.unsqueeze(-1)
            grouped_tensors[-1] = cls.uniform_quantization_threshold(grouped_tensors[-1], cls.OUTLIER_BITS, minval_tensor, maxval_tensor)

            # Quantize outer groups
            for idx in range(len(threshold_lowers) - 1):
                threshold_lower_tensor = torch.tensor(threshold_lowers[idx]).to(input_tensor.device).half()
                threshold_upper_tensor = torch.tensor(threshold_uppers[idx]).to(input_tensor.device).half()

                higher_mask = grouped_tensors[idx] > 0
                lower_mask = grouped_tensors[idx] < 0
                higher_outlier = grouped_tensors[idx] * higher_mask
                lower_outlier = grouped_tensors[idx] * lower_mask

                if use_group_shift:
                    higher_outlier -= threshold_upper_tensor
                    lower_outlier -= threshold_lower_tensor

                if idx == len(threshold_lowers) - 2:
                    total_outlier = cls.uniform_quantization(higher_outlier * higher_mask + lower_outlier * lower_mask, cls.QUANTIZE_BITS)
                else:
                    total_outlier = cls.uniform_quantization(higher_outlier * higher_mask + lower_outlier * lower_mask, cls.OUTLIER_BITS)

                higher_outlier = total_outlier * higher_mask
                lower_outlier = total_outlier * lower_mask
                
                if use_group_shift:
                    higher_outlier += threshold_upper_tensor
                    lower_outlier += threshold_lower_tensor

                grouped_tensors[idx] = (higher_outlier * higher_mask) + (lower_outlier * lower_mask)
        else:
            # Quantize middle_group
            minval_tensor = torch.min(grouped_tensors[-2], dim=-1).values.unsqueeze(-1)
            maxval_tensor = torch.max(grouped_tensors[-2], dim=-1).values.unsqueeze(-1)
            grouped_tensors[-2] = cls.uniform_quantization_threshold(grouped_tensors[-2], cls.QUANTIZE_BITS, minval_tensor, maxval_tensor)

        # heat_map = torch.zeros_like(input_tensor)
        for idx, (tensor, mask) in enumerate(zip(grouped_tensors, masks)):
            result_tensor += tensor * mask
            # heat_map += idx * mask
        heat_map = None

        val_frac = [(torch.count_nonzero(mask) / torch.numel(mask)).item() for mask in masks]
        return (result_tensor, val_frac, heat_map)