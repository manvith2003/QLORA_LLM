import torch
import torch.nn as nn
import math
import sys
sys.path.append('custom_quant')
import custom_quant

class QuantizedLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=16, block_size=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.block_size = block_size
        
        # We don't store the actual W matrix. It will be provided externally during conversion
        # but we need buffers for the quantized weights.
        numel = in_features * out_features
        num_blocks = (numel + self.block_size - 1) // self.block_size
        
        # Buffers for 4-bit quantized backbone weights (frozen)
        self.register_buffer("packed_weights", torch.empty((numel + 1) // 2, dtype=torch.uint8))
        self.register_buffer("block_scales", torch.empty(num_blocks, dtype=torch.float32))
        self.register_buffer("bias_buffer", None) # Optional bias
        
        # LoRA parameters (trainable)
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def quantize_and_store_weights(self, weight, bias=None):
        """Converts an FP32/FP16 weight matrix into our custom 4-bit blocks."""
        with torch.no_grad():
            packed, scales = custom_quant.quantize_4bit(weight, self.block_size)
            self.packed_weights.copy_(packed)
            self.block_scales.copy_(scales)
            if bias is not None:
                if self.bias_buffer is None:
                    self.register_buffer("bias_buffer", bias.clone())
                else:
                    self.bias_buffer.copy_(bias)
                
    def forward(self, x):
        # 1. Dequantize backbone weights on the fly
        W_dequant = custom_quant.dequantize_4bit(
            self.packed_weights, 
            self.block_scales, 
            [self.out_features, self.in_features], 
            self.block_size
        )
        
        # Note: dequantized weight is now on CPU (from our simple C++ implementation).
        W_dequant = W_dequant.to(x.device).to(x.dtype)
        
        # 2. Base model linear projection
        base_out = nn.functional.linear(x, W_dequant, bias=self.bias_buffer)
        
        # 3. LoRA adapter projection
        # x: (..., in_features)
        # lora_A: (r, in_features) -> x @ lora_A.T -> (..., r)
        # lora_B: (out_features, r) -> (..., r) @ lora_B.T -> (..., out_features)
        lora_out = nn.functional.linear(
            nn.functional.linear(x, self.lora_A), 
            self.lora_B
        ) * self.scaling
        
        return base_out + lora_out

def replace_linear_with_qlora(model, module_names_to_replace=None, r=8, lora_alpha=16):
    """Recursively replaces nn.Linear modules with QuantizedLoRALinear."""
    for name, module in dict(model.named_children()).items():
        if isinstance(module, nn.Linear):
            # If a list is provided, only replace if the name matches (e.g. attention projections)
            if module_names_to_replace and not any(m in name for m in module_names_to_replace):
                continue
                
            qlora_layer = QuantizedLoRALinear(
                module.in_features, 
                module.out_features, 
                r=r, 
                lora_alpha=lora_alpha
            )
            # Quantize and store
            qlora_layer.quantize_and_store_weights(module.weight, module.bias)
            
            # Replace
            setattr(model, name, qlora_layer)
        else:
            replace_linear_with_qlora(module, module_names_to_replace, r, lora_alpha)
            
    return model
