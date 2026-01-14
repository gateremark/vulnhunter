# VulnHunter OpenEnv Environment

This directory contains the OpenEnv-compatible environment implementation for VulnHunter.

## Overview

VulnHunter follows the [OpenEnv specification](https://github.com/meta-pytorch/OpenEnv) for RL environments, providing a standard interface for training agents on security vulnerability detection tasks.

## Components

| File | Description |
|------|-------------|
| `server.py` | FastAPI server implementing OpenEnv's `Environment` interface |
| `models.py` | Pydantic models extending OpenEnv's `Action`, `Observation`, `State` |

## OpenEnv Compatibility

The environment inherits from OpenEnv base classes:

```python
from openenv.core.env_server import Environment, Action, Observation, State, create_fastapi_app

class VulnHunterEnv(Environment[VulnHunterAction, VulnHunterObservation, VulnHunterState]):
    def reset(self, seed=None, episode_id=None, **kwargs) -> VulnHunterObservation:
        ...
    
    def step(self, action, timeout_s=None, **kwargs) -> VulnHunterObservation:
        ...
    
    @property
    def state(self) -> VulnHunterState:
        ...
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode |
| `/step` | POST | Execute an action |
| `/state` | GET | Get current environment state |
| `/health` | GET | Health check |
| `/schema` | GET | Get action/observation schemas |

## Running the Server

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn vulnhunter.env_server.server:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t vulnhunter-env .
docker run -p 8000:8000 vulnhunter-env
```

## Configuration

The environment supports the following vulnerability types:
- `sql_injection` (default)
- `xss`
- `path_traversal`

Example reset request:
```bash
curl -X POST http://localhost:8000/reset?vuln_type=sql_injection
```
