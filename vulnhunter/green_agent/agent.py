"""
VulnHunter Green Agent - A2A Compatible Agent Wrapper
Implements the AgentBeats Green Agent protocol for VulnHunter.
"""
from typing import Dict, Any, Optional
import json


class VulnHunterAgent:
    """
    Green Agent wrapper for the VulnHunter security agent.
    
    This agent can analyze code for security vulnerabilities and suggest fixes
    via the A2A (Agent-to-Agent) protocol.
    """
    
    def __init__(self, model_name: str = "gateremark/vulnhunter-agent"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def load_model(self):
        """Lazy-load the model when first needed."""
        if self._loaded:
            return
        
        from unsloth import FastLanguageModel
        import torch
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            self.model_name,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        self._loaded = True
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Analyze code for security vulnerabilities.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Dictionary containing vulnerability analysis and fixes
        """
        self.load_model()
        
        prompt = f"""You are VulnHunter, an AI security researcher. Analyze this code for vulnerabilities:

```python
{code}
```

Identify any vulnerabilities and provide fixes. Format your response as JSON with:
- vulnerability_type (sql_injection, xss, path_traversal, or none)
- severity (high, medium, low)
- description
- fix_suggestion
- fixed_code
"""
        
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # Parse the response
        return {
            "status": "success",
            "analysis": response,
            "model": self.model_name
        }
    
    def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an A2A task request.
        
        Args:
            task: A2A task containing the code to analyze
            
        Returns:
            A2A-compatible response with analysis results
        """
        task_type = task.get("type", "analyze")
        
        if task_type == "analyze":
            code = task.get("code", "")
            if not code:
                return {
                    "status": "error",
                    "error": "No code provided for analysis"
                }
            return self.analyze_code(code)
        
        elif task_type == "capabilities":
            return self.get_capabilities()
        
        else:
            return {
                "status": "error",
                "error": f"Unknown task type: {task_type}"
            }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities for A2A discovery."""
        return {
            "name": "VulnHunter",
            "description": "AI security agent trained with GRPO to detect and fix web vulnerabilities",
            "version": "1.0.0",
            "skills": [
                {
                    "name": "analyze",
                    "description": "Analyze code for security vulnerabilities",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Source code to analyze"}
                        },
                        "required": ["code"]
                    }
                }
            ],
            "supported_vulnerabilities": [
                "sql_injection",
                "xss", 
                "path_traversal"
            ],
            "model": self.model_name
        }


# Singleton instance for the agent
_agent_instance: Optional[VulnHunterAgent] = None


def get_agent() -> VulnHunterAgent:
    """Get or create the VulnHunter agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = VulnHunterAgent()
    return _agent_instance
