"""
VulnHunter OpenEnv Data Models
Defines the Action, Observation, and State types for the environment.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum

class VulnerabilityType(str, Enum):
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    PATH_TRAVERSAL = "path_traversal"
    UNKNOWN = "unknown"

class Action(BaseModel):
    """Actions the security agent can take."""
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

class Observation(BaseModel):
    """What the agent sees after each action."""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    source_code: Optional[str] = None
    current_task: str = ""
    hints: List[str] = Field(default_factory=list)
    
class State(BaseModel):
    """Internal environment state (hidden from agent)."""
    step_count: int = 0
    current_vuln_type: VulnerabilityType = VulnerabilityType.SQL_INJECTION
    vuln_identified: bool = False
    patch_applied: bool = False
    patch_successful: bool = False
    is_terminal: bool = False
    total_reward: float = 0.0

class StepResult(BaseModel):
    """Result of a step in the environment."""
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
