import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qlora_layers import replace_linear_with_qlora
import sys
import copy

def generate_text(model, tokenizer, prompt, max_new_tokens=50):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    model_name = "EleutherAI/pythia-14m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading Base Model (FP32)...")
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat are the main benefits of learning C++?\n\n### Response:\n"
    print("\n--- BASE MODEL OUTPUT ---")
    base_out = generate_text(base_model, tokenizer, prompt)
    print(base_out)
    
    print("\n\nLoading QLoRA Fine-tuned Model (Custom 4-bit)...")
    qlora_model = AutoModelForCausalLM.from_pretrained(model_name)
    qlora_model = replace_linear_with_qlora(qlora_model)
    
    # Load adapters
    try:
        adapter_weights = torch.load("qlora_custom_weights.pt", weights_only=True)
        qlora_model.load_state_dict(adapter_weights, strict=False)
        print("Successfully loaded custom 4-bit QLoRA adapters.")
    except Exception as e:
        print(f"Could not load adapters: {e}")
        
    print("\n--- FINE-TUNED MODEL OUTPUT ---")
    ft_out = generate_text(qlora_model, tokenizer, prompt)
    print(ft_out)

if __name__ == "__main__":
    main()
