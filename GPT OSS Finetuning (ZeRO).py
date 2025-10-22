# run in shell with this command:

"""
!deepspeed --num_gpus=3 "GPT OSS Finetuning (ZeRO).py"
"""

import os

print("Installing required packages...")
os.system("pip install -q datasets transformers peft trl accelerate")
os.system("pip install -q flash-attn --no-build-isolation")

import os
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer
import torch

# Step 1: Load and prepare your dataset
def load_custom_dataset(file_path):
    """Load your JSON dataset and convert to HuggingFace Dataset format"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_data = []
    for conversation in data:
        # Filter out empty messages to avoid training issues
        filtered_messages = []
        for msg in conversation:
            # Skip messages with empty content (except system messages)
            if msg['role'] == 'system' or (msg['content'] and msg['content'].strip()):
                filtered_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        # Only include conversations with meaningful content
        if len(filtered_messages) >= 2:
            formatted_data.append({'messages': filtered_messages})
    
    return Dataset.from_list(formatted_data)

# Load your dataset
dataset = load_custom_dataset('harmony.json')
print(f"✓ Dataset loaded with {len(dataset)} conversations")
print(f"Sample conversation: {dataset[0]}")

# Step 2: Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
print("✓ Tokenizer loaded")

# Step 3: Load model WITHOUT device_map (DeepSpeed will handle device placement)
print("\nLoading model...")
model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-120b",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,  # Use bfloat16 for better stability
    use_cache=False,  # Required for training with gradient checkpointing
    # DO NOT use device_map with DeepSpeed - DeepSpeed handles device placement
)
print("✓ Model loaded")

# Step 4: Configure LoRA for efficient fine-tuning
print("\nConfiguring LoRA...")
lora_config = LoraConfig(
    r=128,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=[
        "self_attn.q_proj",
        "self_attn.k_proj", 
        "self_attn.v_proj",
        "self_attn.o_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

# Wrap the model with LoRA
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# Step 5: Configure training parameters with DeepSpeed ZeRO-3
print("\nConfiguring training...")
training_args = SFTConfig(
    # Training hyperparameters
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    max_ength=256,
    
    # Learning rate schedule
    warmup_ratio=0.03,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    
    # Logging and saving
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    output_dir="gpt-oss-120b-custom-finetuned",
    
    # Optimization settings
    bf16=True,  # Use bfloat16 for better numerical stability
    dataloader_drop_last=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    # Reporting
    report_to="none",  # Change to "wandb" or "tensorboard" if you want logging
    
    # DeepSpeed ZeRO-3 Configuration
    deepspeed="ds_config.json",  # Will create this config file
)

# Create DeepSpeed configuration file
ds_config = {
    "bf16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,  # ZeRO-3: partition optimizer states, gradients, and parameters
        "offload_optimizer": {
            "device": "cpu",  # Offload optimizer states to CPU to save GPU memory
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",  # Offload parameters to CPU (use if needed)
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}

# Save DeepSpeed config
with open("ds_config.json", "w") as f:
    json.dump(ds_config, f, indent=2)
print("✓ DeepSpeed config created: ds_config.json")

# Step 6: Initialize trainer
print("\nInitializing trainer...")
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# Step 7: Start training
print("\n" + "="*50)
print("Starting training...")
print("="*50)
trainer.train()

# Step 8: Save the model
print("\nSaving model...")
trainer.save_model()
print("✓ Training completed and model saved!")

# Cleanup
del trainer
del peft_model
del model
torch.cuda.empty_cache()
import gc
gc.collect()

print("\n" + "="*50)
print("Training complete!")
print("="*50)

# Step 9: Test inference (optional)
def test_inference(prompt="Tell me a story about a robot."):
    """Test the fine-tuned model with a sample prompt"""
    print(f"\nTesting inference with prompt: '{prompt}'")
    
    # Load the base model for inference
    base_model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-120b",
        torch_dtype=torch.bfloat16,
        device_map="auto",  # OK to use device_map for inference
        use_cache=True
    )
    
    # Load the fine-tuned LoRA weights
    model = PeftModel.from_pretrained(
        base_model, 
        "gpt-oss-120b-custom-finetuned"
    )
    model = model.merge_and_unload()
    
    # Prepare the prompt
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    gen_kwargs = {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    print("Generating response...")
    output_ids = model.generate(input_ids, **gen_kwargs)
    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    print("\n" + "="*50)
    print("Model response:")
    print("="*50)
    print(response)
    print("="*50)
    
    # Cleanup
    del base_model
    del model
    torch.cuda.empty_cache()
    gc.collect()

# Uncomment to test inference
# test_inference("What is machine learning?")