# VulnHunter

**An AI Security Agent trained with Reinforcement Learning to find and patch vulnerabilities.**

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/gateremark/vulnhunter-agent)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Environment-blue)](https://github.com/meta-pytorch/OpenEnv)
[![AgentBeats](https://img.shields.io/badge/AgentBeats-Competition-green)](https://rdi.berkeley.edu/agentx-agentbeats)

## 🎯 Overview

VulnHunter is an OpenEnv-compatible reinforcement learning environment for training AI agents to:
- **Detect** security vulnerabilities (SQL Injection, XSS, Path Traversal)
- **Locate** vulnerable code patterns
- **Generate** secure patches

Built for the [AgentX-AgentBeats OpenEnv Challenge](https://rdi.berkeley.edu/agentx-agentbeats).

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│          VulnHunter OpenEnv Environment          │
├─────────────────────────────────────────────────┤
│  State: Vulnerable web application code          │
│  Actions: read_file, command, identify, patch    │
│  Rewards: +1.0 successful patch, +0.3 identify   │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│           Unsloth + GRPO Training               │
│  • Qwen2.5-Coder-7B-Instruct                   │
│  • 4-bit quantization (QLoRA)                   │
│  • Group Relative Policy Optimization           │
└─────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Install Dependencies
```bash
pip install unsloth trl fastapi uvicorn pydantic
```

### Run the Environment
```bash
cd vulnhunter
uvicorn vulnhunter.env_server.server:app --host 0.0.0.0 --port 8000
```

### Use the Trained Model
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "gateremark/vulnhunter-agent"
)

prompt = """Analyze this code for vulnerabilities:
@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

## 📁 Project Structure

```
vulnhunter/
├── vulnhunter/
│   ├── env_server/        # OpenEnv implementation
│   │   ├── models.py      # Action/Observation/State
│   │   └── server.py      # FastAPI server
│   ├── vulnerable_app/    # Target vulnerable Flask app
│   │   └── app.py
│   ├── green_agent/       # AgentBeats evaluator
│   └── training/          # GRPO training scripts
├── Dockerfile             # Environment containerization
└── requirements.txt
```

---

### **Track**: Reinforcement Learning 

---

## 📚 Resources

- [OpenEnv Documentation](https://meta-pytorch.org/OpenEnv/)
- [Unsloth AI](https://github.com/unslothai/unsloth)
- [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl)
- [AgentBeats Competition](https://rdi.berkeley.edu/agentx-agentbeats)

---

Built for the AgentBeats Competition 2026
