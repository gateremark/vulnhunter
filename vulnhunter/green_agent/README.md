# VulnHunter Green Agent

A2A-compatible Green Agent wrapper for the VulnHunter security analysis agent.

## Overview

This Green Agent exposes the VulnHunter model via the A2A (Agent-to-Agent) protocol, allowing it to be discovered and used by other agents in the AgentBeats ecosystem.

## Running the Agent

### Local
```bash
pip install -r requirements.txt
python server.py
```

### Docker
```bash
docker build -t vulnhunter-agent .
docker run -p 9009:9009 vulnhunter-agent
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/.well-known/agent.json` | GET | A2A Agent Card for discovery |
| `/agent/task` | POST | Handle A2A task requests |
| `/capabilities` | GET | Get agent capabilities |
| `/analyze` | POST | Direct vulnerability analysis |

## Agent Card

The agent exposes its capabilities via the standard A2A agent card:

```json
{
  "name": "VulnHunter",
  "description": "AI security agent that detects and fixes web vulnerabilities",
  "skills": [{
    "id": "analyze_code",
    "name": "Analyze Code",
    "description": "Analyze source code for security vulnerabilities"
  }]
}
```

## Example Usage

```python
import requests

# Analyze code for vulnerabilities
response = requests.post("http://localhost:9009/agent/task", json={
    "type": "analyze",
    "code": '''
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    '''
})

print(response.json())
```

## Supported Vulnerabilities

- SQL Injection
- Cross-Site Scripting (XSS)
- Path Traversal

## Links

- Model: [gateremark/vulnhunter-agent](https://huggingface.co/gateremark/vulnhunter-agent)
- GitHub: [github.com/gateremark/vulnhunter](https://github.com/gateremark/vulnhunter)
