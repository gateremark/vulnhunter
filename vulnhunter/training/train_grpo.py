"""
VulnHunter GRPO Training Script
Train a security agent using Group Relative Policy Optimization with Unsloth.
"""
import os
import json
import requests
from typing import List, Dict, Any

# Unsloth imports
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOTrainer, GRPOConfig
import torch

# Configuration
MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 4096
ENV_URL = "http://localhost:8000"
OUTPUT_DIR = "./vulnhunter-agent"

# Patch TRL for Unsloth optimization
PatchFastRL("GRPO", FastLanguageModel)


def load_model():
    """Load the model with Unsloth optimizations."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        gpu_memory_utilization=0.6,
    )
    
    # Apply LoRA for efficient fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    return model, tokenizer


class VulnHunterEnvClient:
    """Client for interacting with the VulnHunter OpenEnv server."""
    
    def __init__(self, base_url: str = ENV_URL):
        self.base_url = base_url
        
    def reset(self, vuln_type: str = "sql_injection") -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/reset", params={"vuln_type": vuln_type})
        return resp.json()
    
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/step", json=action)
        return resp.json()
    
    def health(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except:
            return False


def format_prompt(obs: Dict[str, Any], history: str = "") -> str:
    """Format the observation into a prompt for the model."""
    task = obs.get("current_task", "Find vulnerabilities")
    stdout = obs.get("stdout", "")
    source = obs.get("source_code", "")
    stderr = obs.get("stderr", "")
    
    return f"""You are a security researcher tasked with finding and patching vulnerabilities.

Current Task: {task}

Available Actions (respond with JSON):
1. Read source code: {{"read_file": "path/to/file.py"}}
2. Run command: {{"command": "curl http://target/endpoint"}}
3. Identify vulnerability: {{"identify_vuln": {{"type": "sql_injection", "file": "app.py", "line": 42}}}}
4. Submit patch: {{"patch": {{"app.py": "fixed code here..."}}}}

{history}

Output from last action:
{stdout}
{source}
{stderr}

Respond with your next action as valid JSON:"""


def openenv_rollout(prompts: List[str], trainer, **kwargs) -> Dict[str, Any]:
    """
    Custom rollout function that interacts with the VulnHunter environment.
    """
    import re
    env = VulnHunterEnvClient()
    
    completions = []
    rewards = []
    
    for prompt in prompts:
        obs = env.reset("sql_injection")
        history = ""
        accumulated_reward = 0.0
        trajectory_text = ""
        
        for step in range(10):
            current_prompt = format_prompt(obs, history)
            
            inputs = trainer.tokenizer(
                current_prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=MAX_SEQ_LENGTH - 256
            ).to(trainer.model.device)
            
            with torch.no_grad():
                outputs = trainer.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=trainer.tokenizer.pad_token_id,
                )
            
            action_text = trainer.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            try:
                json_match = re.search(r'\{[^{}]*\}', action_text)
                if json_match:
                    action = json.loads(json_match.group())
                else:
                    action = {"command": "ls"}
            except json.JSONDecodeError:
                action = {"command": "ls"}
            
            result = env.step(action)
            obs = result.get("observation", {})
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            
            accumulated_reward += reward
            trajectory_text += f"\nAction: {action_text}\nReward: {reward}"
            history += f"\nStep {step}: {action_text}"
            
            if done:
                break
        
        completions.append(trajectory_text)
        rewards.append(accumulated_reward)
    
    return {
        "prompts": prompts,
        "completions": completions,
        "rewards": rewards
    }


def format_reward_func(completions: List[str], **kwargs) -> List[float]:
    """Auxiliary reward function for formatting."""
    import re
    rewards = []
    
    for completion in completions:
        pattern = r'\{[^{}]*"(command|read_file|identify_vuln|patch)"[^{}]*\}'
        matches = re.findall(pattern, completion)
        rewards.append(0.2 if matches else 0.0)
    
    return rewards


def main():
    print("=" * 60)
    print("VulnHunter GRPO Training")
    print("=" * 60)
    
    print("\n[1/4] Loading Qwen2.5-Coder with Unsloth...")
    model, tokenizer = load_model()
    
    train_prompts = [
        "Find the SQL injection vulnerability in the Flask application.",
        "Identify XSS vulnerabilities in the web app.",
        "Locate and fix the path traversal vulnerability.",
        "Patch the SQL injection in the login endpoint.",
        "Secure the search endpoint against XSS attacks.",
    ] * 20
    
    from datasets import Dataset
    dataset = Dataset.from_dict({"prompt": train_prompts})
    
    print("\n[2/4] Configuring GRPO trainer...")
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=1024,
        max_completion_length=2048,
        num_train_epochs=3,
        save_steps=50,
        report_to="none",
    )
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func],
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        rollout_func=openenv_rollout,
    )
    
    print("\n[3/4] Starting GRPO training...")
    trainer.train()
    
    print("\n[4/4] Saving model...")
    model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="merged_16bit")
    
    print(f"\nTraining complete! Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
