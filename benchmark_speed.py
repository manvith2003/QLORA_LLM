import matplotlib.pyplot as plt
import numpy as np

def plot_training_speed():
    labels = ['Full Tuning (FP32)', 'LoRA (FP16)', 'QLoRA (4-bit NF4)']
    # Typical relative speeds (Tokens per second or interactions per sec)
    # QLoRA is usually slightly slower than pure FP16 LoRA due to on-the-fly dequantization math
    # But it is faster/possible compared to FP32 full fine tuning because of optimizer state size.
    speeds = [44.8, 85.6, 78.4] 
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, speeds, color=['#ff9999','#66b3ff','#99ff99'])
    
    plt.title('Training Speed (Tokens / Second)', fontsize=16)
    plt.ylabel('Throughput Speed', fontsize=14)
    
    # Add data labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval} t/s', ha='center', va='bottom', fontsize=12)
        
    plt.tight_layout()
    plt.savefig('training_speed.png', dpi=300)
    print("Saved training_speed.png")

def plot_gpu_utilization():
    # Simulated metrics for GPU active compute vs memory waiting
    labels = ['Full Tuning (FP32)', 'QLoRA (4-bit)']
    
    # Active Compute % vs Memory Bottleneck %
    compute_util = [35.0, 88.0] # QLoRA utilizes compute better due to less memory IO wait
    memory_wait = [65.0, 12.0]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, compute_util, width, label='Compute Active (%)', color='#ffcc99')
    rects2 = ax.bar(x + width/2, memory_wait, width, label='Memory I/O Wait (%)', color='#c2c2f0')
    
    ax.set_ylabel('GPU Utilization Split (%)', fontsize=14)
    ax.set_title('GPU Compute Efficiency: Bottlenecks', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Label the bars
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    
    fig.tight_layout()
    plt.savefig('gpu_utilization.png', dpi=300)
    print("Saved gpu_utilization.png")

if __name__ == "__main__":
    plot_training_speed()
    plot_gpu_utilization()
