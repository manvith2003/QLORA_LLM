import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from qlora_layers import replace_linear_with_qlora

def main():
    print("Loading base model GPT-2 for demonstration...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in FP32 format originally
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Replace linear layers (e.g., attention projections) with our custom 4-bit C++ QLoRA layer
    # For GPT-2, linear layers using Conv1D can be tricky, but recent HuggingFace transformers
    # often use nn.Linear in some architectures, however GPT2 officially uses Conv1D. 
    # Wait, GPT2 uses Conv1D, not nn.Linear. Let's convert Conv1D to nn.Linear for compatibility 
    # or just use a modern tiny model like "EleutherAI/pythia-14m" which uses standard nn.Linear.
    
    model_name = "EleutherAI/pythia-14m" # Or "pythia-70m"
    print(f"Switching to {model_name} as it uses standard nn.Linear layers for simpler replacement.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print("Initial model memory footprint:")
    model_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    print(f"{model_mem:.2f} MB")
    
    # We will replace all standard Linear layers with our 4-bit C++ quant version + LoRA
    print("Applying Custom QLoRA to standard linear layers...")
    model = replace_linear_with_qlora(model)
    
    print("Post-QLoRA model memory footprint (Frozen Backbone is now 4-bit!):")
    # Tally up buffers (quantized weights) + active parameters
    buffer_mem = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024**2)
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    print(f"Buffers (4-bit Weights & Scales): {buffer_mem:.2f} MB")
    print(f"Trainable Parameters (LoRA): {param_mem:.2f} MB")
    print(f"Total: {buffer_mem + param_mem:.2f} MB")
    
    # Freeze all non-LoRA parameters (like embeddings, layernorm) for strict QLoRA
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            trainable_params += param.numel()
        total_params += param.numel()
        
    print(f"Trainable Parameters: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")

    # Load toy dataset (Alpaca)
    print("Loading Dataset subset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:1%]") # Use 1% for fast demo
    
    def tokenize_function(examples):
        # Alpaca prompt formatting
        prompts = [
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{inst}\n\n### Response:\n{out}" 
            for inst, out in zip(examples["instruction"], examples["output"])
        ]
        return tokenizer(prompts, padding="max_length", truncation=True, max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])
    
    dataloader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    # Training Loop
    print("Starting Training Loop (Custom C++ 4-bit Dequant on-the-fly)...")
    model.train()
    epochs = 1
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(dataloader):.4f}")
        
    print("Saving fine-tuned LoRA weights...")
    # Just save the trainable LoRA adapters
    lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state_dict, "qlora_custom_weights.pt")
    print("Done!")

if __name__ == '__main__':
    main()
