import torch
import sys
sys.path.append('custom_quant')
import custom_quant

def test():
    weights = torch.randn(10, 10, dtype=torch.float32)
    
    # Quantize
    packed, scales = custom_quant.quantize_4bit(weights, 2)
    
    # Dequantize
    dequantized = custom_quant.dequantize_4bit(packed, scales, weights.shape, 2)
    
    print("Original Weights:")
    print(weights[0, :5])
    print("Dequantized Weights:")
    print(dequantized[0, :5])
    
    mse = torch.nn.functional.mse_loss(weights, dequantized)
    print(f"Mean Squared Error: {mse.item():.6f}")

if __name__ == '__main__':
    test()
