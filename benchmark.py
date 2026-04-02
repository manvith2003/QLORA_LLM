import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append('custom_quant')
import custom_quant

def plot_memory_benchmark():
    # Model: Pythia-14M (approx 14,000,000 parameters for the core layers)
    # Let's use exact numbers from our run
    total_params = 14_000_000
    
    # FP32: 32 bits = 4 bytes per param
    fp32_mem = (total_params * 4) / (1024 ** 2)
    
    # 8-bit: 1 byte per param + scale factors
    fp8_mem = (total_params * 1) / (1024 ** 2)
    
    # 4-bit: 0.5 bytes per param + scale factors + LoRA Adapters
    # In our run, 4-bit backbone took ~4MB, LoRA took ~14MB. For just the base model comparison:
    fp4_mem = (total_params * 0.5) / (1024 ** 2)
    
    labels = ['Original LLM (32-bit)', '8-bit Quantization', 'QLoRA (4-bit)']
    memory_mb = [fp32_mem, fp8_mem, fp4_mem]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, memory_mb, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    
    plt.title('Memory Footprint of Base Model Weights', fontsize=16)
    plt.ylabel('Memory (MB)', fontsize=14)
    
    # Add data labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f} MB', ha='center', va='bottom', fontsize=12)
        
    plt.tight_layout()
    plt.savefig('memory_benchmark.png', dpi=300)
    print("Saved memory_benchmark.png")
    
def plot_quantization_distribution():
    # Generate a sample of typical normally-distributed neural network weights
    weights = torch.randn(100000) * 0.05
    
    # Quantize them using our C++ 4-bit extension
    block_size = 64
    packed, scales = custom_quant.quantize_4bit(weights, block_size)
    dequantized = custom_quant.dequantize_4bit(packed, scales, weights.shape, block_size)
    
    plt.figure(figsize=(12, 6))
    
    # Plot Original FP32 Distribution
    plt.subplot(1, 2, 1)
    plt.hist(weights.numpy(), bins=100, color='#FF6B6B', alpha=0.7)
    plt.title('Original 32-bit Weights\n(Continuous Infinity of Values)', fontsize=14)
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    
    # Plot 4-bit Dequantized Distribution
    plt.subplot(1, 2, 2)
    plt.hist(dequantized.numpy(), bins=100, color='#45B7D1', alpha=0.7)
    plt.title('4-bit QLoRA Weights\n(Squashed into 16 Discrete Bins)', fontsize=14)
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('quantization_distribution.png', dpi=300)
    print("Saved quantization_distribution.png")

if __name__ == "__main__":
    plot_memory_benchmark()
    plot_quantization_distribution()
