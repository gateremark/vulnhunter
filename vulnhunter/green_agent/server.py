"""
VulnHunter Green Agent - A2A Server
Exposes the VulnHunter agent via HTTP with A2A protocol support.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn

from agent import get_agent

app = FastAPI(
    title="VulnHunter Green Agent",
    description="A2A-compatible security vulnerability detection agent",
    version="1.0.0"
)


# Agent Card - A2A Discovery
AGENT_CARD = {
    "name": "VulnHunter",
    "description": "AI security agent that detects and fixes web application vulnerabilities using GRPO-trained models",
    "url": "http://localhost:9009",
    "version": "1.0.0",
    "skills": [
        {
            "id": "analyze_code",
            "name": "Analyze Code",
            "description": "Analyze source code for security vulnerabilities like SQL injection, XSS, and path traversal",
            "tags": ["security", "code-analysis", "vulnerability-detection"],
            "input_modes": ["text"],
            "output_modes": ["text", "json"]
        }
    ],
    "default_input_modes": ["text"],
    "default_output_modes": ["text"],
    "capabilities": {
        "streaming": False,
        "push_notifications": False
    }
}


class TaskRequest(BaseModel):
    """A2A Task Request"""
    type: str = "analyze"
    code: Optional[str] = None
    message: Optional[str] = None


class TaskResponse(BaseModel):
    """A2A Task Response"""
    status: str
    analysis: Optional[str] = None
    error: Optional[str] = None
    model: Optional[str] = None


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "healthy", "agent": "VulnHunter"}


@app.get("/.well-known/agent.json")
def agent_card():
    """
    A2A Agent Card endpoint.
    Returns agent metadata for discovery by other agents.
    """
    return AGENT_CARD


@app.get("/agent/card")
def agent_card_alt():
    """Alternative agent card endpoint."""
    return AGENT_CARD


@app.post("/agent/task")
async def handle_task(request: TaskRequest) -> Dict[str, Any]:
    """
    Handle an A2A task request.
    
    Supports:
    - analyze: Analyze code for vulnerabilities
    - capabilities: Return agent capabilities
    """
    agent = get_agent()
    
    task = {
        "type": request.type,
        "code": request.code or request.message
    }
    
    result = agent.handle_task(task)
    return result


@app.get("/capabilities")
def capabilities():
    """Return agent capabilities."""
    agent = get_agent()
    return agent.get_capabilities()


@app.post("/analyze")
async def analyze(code: str) -> Dict[str, Any]:
    """
    Direct endpoint to analyze code.
    
    Args:
        code: Source code to analyze for vulnerabilities
    """
    agent = get_agent()
    return agent.analyze_code(code)


if __name__ == "__main__":
    print("Starting VulnHunter Green Agent on port 9009...")
    print("Agent Card: http://localhost:9009/.well-known/agent.json")
    uvicorn.run(app, host="0.0.0.0", port=9009)
