"""
╔══════════════════════════════════════════════════════════════════╗
║   QLoRA LIVE DEMO — Base Model vs Fine-Tuned (4-bit Custom C++) ║
║   Authors: Manvith M & Shubhendu Shukla                        ║
╚══════════════════════════════════════════════════════════════════╝
Run this script during your presentation to show the implementation working.
Usage: python3 live_demo.py
"""

import torch
import torch.nn as nn
import time
import sys
import os

sys.path.append('custom_quant')
import custom_quant
from transformers import AutoModelForCausalLM, AutoTokenizer
from qlora_layers import replace_linear_with_qlora

# ─── Pretty Printing Helpers ───
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
BOLD    = "\033[1m"
RESET   = "\033[0m"
LINE    = "═" * 65

def banner(text):
    print(f"\n{CYAN}{LINE}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{CYAN}{LINE}{RESET}\n")

def section(text):
    print(f"\n{YELLOW}▶ {text}{RESET}")
    print(f"{YELLOW}{'─' * 50}{RESET}")

def success(text):
    print(f"  {GREEN}✓ {text}{RESET}")

def metric(label, value):
    print(f"  {BOLD}{label:<35}{RESET} {value}")

# ─── Configuration ───
MODEL_NAME = "EleutherAI/pythia-14m"
PROMPTS = [
    "What are the main benefits of learning C++?",
    "Explain what machine learning is in simple terms.",
    "How does the internet work?",
]

def get_memory_mb(model):
    """Calculate total memory of a model in MB."""
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    buffer_mem = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024**2)
    return param_mem, buffer_mem

def generate(model, tokenizer, prompt, max_tokens=80):
    """Generate text from a prompt."""
    device = next(model.parameters()).device
    full_prompt = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{prompt}\n\n### Response:\n"
    )
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.2,
        )
    elapsed = time.time() - start
    
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tokens_generated = len(generated_ids)
    return text.strip(), elapsed, tokens_generated

def main():
    banner("QLoRA LIVE DEMONSTRATION")
    print(f"  Model       : {MODEL_NAME}")
    print(f"  Device      : {'Apple MPS (GPU)' if torch.backends.mps.is_available() else 'CPU'}")
    print(f"  Quantization: Custom C++ 4-bit (from scratch)")
    print()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # ══════════════════════════════════════════════
    # STEP 1: Load Tokenizer
    # ══════════════════════════════════════════════
    section("STEP 1: Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    success("Tokenizer loaded.")

    # ══════════════════════════════════════════════
    # STEP 2: Load Base Model (Standard 32-bit)
    # ══════════════════════════════════════════════
    section("STEP 2: Loading BASE Model (Full 32-bit Precision)")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    base_model.to(device)
    base_model.eval()

    param_mem, buf_mem = get_memory_mb(base_model)
    total_base = param_mem + buf_mem
    success(f"Base model loaded.")
    metric("Parameters Memory:", f"{param_mem:.2f} MB")
    metric("Total Footprint:", f"{total_base:.2f} MB")

    # ══════════════════════════════════════════════
    # STEP 3: Load QLoRA Model (Custom 4-bit C++)
    # ══════════════════════════════════════════════
    section("STEP 3: Loading QLoRA Model (Custom 4-bit C++ Quantization)")
    qlora_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    print(f"  → Performing model surgery (replacing nn.Linear → QuantizedLoRALinear)...")
    qlora_model = replace_linear_with_qlora(qlora_model)
    
    # Load fine-tuned LoRA adapter weights
    weights_path = "qlora_custom_weights.pt"
    if os.path.exists(weights_path):
        adapter_weights = torch.load(weights_path, weights_only=True, map_location="cpu")
        qlora_model.load_state_dict(adapter_weights, strict=False)
        success(f"Loaded fine-tuned LoRA adapters from {weights_path}")
    else:
        print(f"  {RED}⚠ No fine-tuned weights found. Using freshly quantized model.{RESET}")
    
    qlora_model.to(device)
    qlora_model.eval()

    param_mem_q, buf_mem_q = get_memory_mb(qlora_model)
    total_qlora = param_mem_q + buf_mem_q
    success(f"QLoRA model loaded.")
    metric("Frozen Backbone (4-bit):", f"{buf_mem_q:.2f} MB")
    metric("LoRA Adapters (32-bit):", f"{param_mem_q:.2f} MB")
    metric("Total Footprint:", f"{total_qlora:.2f} MB")

    # ══════════════════════════════════════════════
    # STEP 4: Memory Comparison
    # ══════════════════════════════════════════════
    section("STEP 4: MEMORY COMPARISON")
    reduction = ((total_base - total_qlora) / total_base) * 100
    metric("Base Model (FP32):", f"{total_base:.2f} MB")
    metric("QLoRA Model (4-bit + LoRA):", f"{total_qlora:.2f} MB")
    metric("Memory Saved:", f"{GREEN}{reduction:.1f}%{RESET}")

    # ══════════════════════════════════════════════
    # STEP 5: Quantization Accuracy Check
    # ══════════════════════════════════════════════
    section("STEP 5: C++ Quantization Accuracy Check")
    test_tensor = torch.randn(1024)
    packed, scales = custom_quant.quantize_4bit(test_tensor, 64)
    recovered = custom_quant.dequantize_4bit(packed, scales, test_tensor.shape, 64)
    mse = torch.mean((test_tensor - recovered) ** 2).item()
    metric("Test Tensor Size:", "1024 values")
    metric("Mean Squared Error:", f"{mse:.6f}")
    metric("Quality:", f"{GREEN}{'Excellent' if mse < 0.01 else 'Acceptable'}{RESET}")

    # ══════════════════════════════════════════════
    # STEP 6: Side-by-Side Text Generation
    # ══════════════════════════════════════════════
    section("STEP 6: SIDE-BY-SIDE TEXT GENERATION")
    
    for i, prompt in enumerate(PROMPTS):
        print(f"\n  {BOLD}Prompt {i+1}:{RESET} \"{prompt}\"")
        print(f"  {'─' * 55}")
        
        # Base model
        base_text, base_time, base_tokens = generate(base_model, tokenizer, prompt)
        base_speed = base_tokens / base_time if base_time > 0 else 0
        
        print(f"  {RED}[BASE MODEL — 32-bit]{RESET}")
        print(f"  {base_text[:200]}{'...' if len(base_text) > 200 else ''}")
        print(f"  ⏱ {base_time:.2f}s | {base_tokens} tokens | {base_speed:.1f} tok/s")
        
        # QLoRA model
        qlora_text, qlora_time, qlora_tokens = generate(qlora_model, tokenizer, prompt)
        qlora_speed = qlora_tokens / qlora_time if qlora_time > 0 else 0
        
        print(f"\n  {GREEN}[QLoRA MODEL — 4-bit Custom C++]{RESET}")
        print(f"  {qlora_text[:200]}{'...' if len(qlora_text) > 200 else ''}")
        print(f"  ⏱ {qlora_time:.2f}s | {qlora_tokens} tokens | {qlora_speed:.1f} tok/s")
        print()

    # ══════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════
    banner("SUMMARY")
    print(f"  ┌────────────────────────┬──────────────┬──────────────┐")
    print(f"  │ {BOLD}Metric{RESET}                 │ {RED}Base (FP32){RESET}  │ {GREEN}QLoRA (4-bit){RESET}│")
    print(f"  ├────────────────────────┼──────────────┼──────────────┤")
    print(f"  │ Memory Footprint       │ {total_base:>8.2f} MB  │ {total_qlora:>8.2f} MB  │")
    print(f"  │ Memory Reduction       │      —       │ {reduction:>8.1f} %   │")
    print(f"  │ Quantization MSE       │      —       │   {mse:>8.6f} │")
    print(f"  │ Custom C++ Engine      │      No      │     {GREEN}Yes{RESET}      │")
    print(f"  └────────────────────────┴──────────────┴──────────────┘")
    print()
    print(f"  {GREEN}{BOLD}✓ All benchmarks completed successfully.{RESET}")
    print(f"  {CYAN}  Built from scratch with custom C++ quantization.{RESET}")
    print()

if __name__ == "__main__":
    main()
