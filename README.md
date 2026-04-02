# QLoRA - Memory-Efficient LLM Fine-Tuning (From Scratch)

This project is a custom, from-scratch implementation of **QLoRA (Quantized Low-Rank Adaptation)** for fine-tuning Large Language Models (LLMs) on consumer-grade hardware.

Instead of relying on high-level libraries, this repository features a **Custom C++ PyTorch Extension** to handle 4-bit blockwise quantization and memory packing, bypassing traditional hardware barriers.

## 🚀 Key Features

- **Custom 4-Bit Quantization:** Native C++ implementation of weight packing and unpacking to achieve **~88% memory reduction**.
- **Low-Rank Adaptation (LoRA):** Implementation of $W' = W + BA$ logic to train only $<1\%$ of model parameters.
- **Surgical Model Replacement:** Tools to automatically replace standard `nn.Linear` layers with custom `QuantizedLoRALinear` modules.
- **High Efficiency:** Benchmarked to show up to **2x faster** training throughput and **88% GPU compute utilization**.

## 📊 Performance Benchmark

| Metric | Original (32-bit) | QLoRA (4-bit) | Improvement |
| :--- | :--- | :--- | :--- |
| **Model Memory** | 53.4 MB | **6.7 MB** | **87.5% Savings** |
| **Training Speed** | 45.2 t/s | **78.4 t/s** | **~1.73x Faster** |
| **GPU Efficiency** | 35% | **88%** | **Reduced I/O Wait** |

### Quantization Quality
- **Weight Recovery MSE:** `0.001712` (Negligible precision loss).

## 🛠️ Project Structure

- `custom_quant/`: C++ source code and PyTorch extension setup.
- `qlora_layers.py`: Custom PyTorch layers for quantized LoRA.
- `train_qlora.py`: Training pipeline for instruction tuning.
- `benchmark.py` & `benchmark_speed.py`: Performance measurement tools.
- `qlora_presentation.tex`: Comprehensive technical documentation and slides.

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/manvith2003/QLORA_LLM.git
   cd QLORA_LLM
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Build the custom C++ extension:
   ```bash
   cd custom_quant
   python3 setup.py install
   cd ..
   ```

## 📜 Acknowledgments
Developed by **Manvith M** and **Shubhendu Shukla** as a deep dive into memory-efficient machine learning.
