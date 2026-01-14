"""
VulnHunter OpenEnv Data Models
Defines the Action, Observation, and State types following OpenEnv base classes.
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum

# Add OpenEnv to path (assumes openenv is installed or in PYTHONPATH)
# For development, you can add: sys.path.insert(0, "/path/to/OpenEnv/src")
try:
    from openenv.core.env_server import Action as BaseAction
    from openenv.core.env_server import Observation as BaseObservation
    from openenv.core.env_server import State as BaseState
except ImportError:
    # Fallback for environments without openenv installed
    from pydantic import BaseModel, Field
    
    class BaseAction(BaseModel):
        metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class BaseObservation(BaseModel):
        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class BaseState(BaseModel):
        episode_id: Optional[str] = None
        step_count: int = 0

from pydantic import Field


class VulnerabilityType(str, Enum):
    """Types of security vulnerabilities supported by VulnHunter."""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    PATH_TRAVERSAL = "path_traversal"
    UNKNOWN = "unknown"


class VulnHunterAction(BaseAction):
    """Actions the security agent can take.
    
    Inherits from OpenEnv's base Action class.
    """
    # Option 1: Run a command (e.g., curl, sqlmap)
    command: Optional[str] = Field(None, description="Shell command to execute")
    # Option 2: Read a source file
    read_file: Optional[str] = Field(None, description="Path to source file to read")
    # Option 3: Submit a patch
    patch: Optional[Dict[str, str]] = Field(None, description="File path -> patched content")
    # Option 4: Identify vulnerability
    identify_vuln: Optional[Dict[str, str]] = Field(
        None, 
        description="Vulnerability identification: {type, file, line}"
    )
    # Terminate the episode
    terminate: bool = Field(False, description="End the episode")


class VulnHunterObservation(BaseObservation):
    """What the agent sees after each action.
    
    Inherits from OpenEnv's base Observation class.
    Includes done, reward, metadata from base class.
    """
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    source_code: Optional[str] = None
    current_task: str = ""
    hints: List[str] = Field(default_factory=list)


class VulnHunterState(BaseState):
    """Internal environment state.
    
    Inherits from OpenEnv's base State class.
    Includes step_count, episode_id from base class.
    """
    current_vuln_type: VulnerabilityType = VulnerabilityType.SQL_INJECTION
    vuln_identified: bool = False
    patch_applied: bool = False
    patch_successful: bool = False
    is_terminal: bool = False
    total_reward: float = 0.0


# Aliases for backwards compatibility
Action = VulnHunterAction
Observation = VulnHunterObservation
State = VulnHunterState


class StepResult:
    """Result of a step in the environment."""
    def __init__(
        self,
        observation: VulnHunterObservation,
        reward: float,
        done: bool,
        info: Optional[Dict[str, Any]] = None
    ):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info or {}
