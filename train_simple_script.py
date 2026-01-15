"""
VulnHunter Simplified Training Script
Uses GRPO with a simulated environment for faster training.
"""
import os
import json
import re
from typing import List, Dict, Any
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Unsloth imports
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOTrainer, GRPOConfig
import torch
from datasets import Dataset

# Configuration
MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "./vulnhunter-agent"

print("=" * 60)
print("VulnHunter GRPO Training")
print("=" * 60)

# Patch TRL for Unsloth
PatchFastRL("GRPO", FastLanguageModel)

print("\n[1/4] Loading Qwen2.5-Coder with Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print("Model loaded and LoRA applied!")

# Create training prompts
SYSTEM_PROMPT = """You are VulnHunter, an AI security researcher. Your task is to find and patch security vulnerabilities in web applications.

When analyzing code, look for:
- SQL Injection: Unsanitized input in SQL queries
- XSS: Unescaped user input in HTML
- Path Traversal: Unchecked file paths

Respond with your analysis and a JSON action like:
{"identify_vuln": {"type": "sql_injection", "file": "app.py", "line": 45}}
"""

VULNERABLE_CODE_SQL = '''
@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    cursor.execute(query)
    return "OK" if cursor.fetchone() else "Failed"
'''

VULNERABLE_CODE_XSS = '''
@app.route("/search")
def search():
    query = request.args.get("q", "")
    return f"<h1>Results for: {query}</h1>"
'''

train_prompts = [
    f"{SYSTEM_PROMPT}\n\nAnalyze this code:\n```python{VULNERABLE_CODE_SQL}```\n\nWhat vulnerability exists?",
    f"{SYSTEM_PROMPT}\n\nThis has SQL injection. How to fix?\n```python{VULNERABLE_CODE_SQL}```",
    f"{SYSTEM_PROMPT}\n\nAnalyze:\n```python{VULNERABLE_CODE_XSS}```\n\nIdentify the security issue.",
    f"{SYSTEM_PROMPT}\n\nFix the XSS vulnerability:\n```python{VULNERABLE_CODE_XSS}```",
] * 25

dataset = Dataset.from_dict({"prompt": train_prompts})
print(f"Dataset size: {len(dataset)}")


def security_reward_func(completions: List[str], **kwargs) -> List[float]:
    """Reward function for security analysis quality."""
    rewards = []
    for c in completions:
        reward = 0.0
        c_lower = c.lower()
        
        # Reward correct identification
        if "sql injection" in c_lower or "sql_injection" in c_lower:
            reward += 0.5
        if "xss" in c_lower or "cross-site" in c_lower:
            reward += 0.4
        if "parameterized" in c_lower or "prepared" in c_lower:
            reward += 0.3
        if "escape" in c_lower or "sanitize" in c_lower:
            reward += 0.3
        
        # Reward proper JSON format
        if re.search(r'\{[^{}]*"identify_vuln"', c):
            reward += 0.3
        if re.search(r'\{[^{}]*"patch"', c):
            reward += 0.2
            
        rewards.append(min(reward, 1.5))
    return rewards


print("\n[2/4] Configuring GRPO trainer...")
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_prompt_length=512,
    max_completion_length=512,
    num_train_epochs=2,
    save_steps=20,
    report_to="none",
    bf16=True,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[security_reward_func],
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

print("\n[3/4] Starting GRPO training...")
trainer.train()

print("\n[4/4] Saving model...")
model.save_pretrained_merged(
    OUTPUT_DIR,
    tokenizer,
    save_method="merged_16bit",
)

# Push to HuggingFace
print("\nPushing to HuggingFace Hub...")
model.push_to_hub_merged(
    "gateremark/vulnhunter-agent",
    tokenizer,
    save_method="merged_16bit",
    private=False,
)

print("\n" + "=" * 60)
print("Training complete!")
print(f"Model saved locally: {OUTPUT_DIR}")
print("Model on HF: huggingface.co/gateremark/vulnhunter-agent")
print("=" * 60)
