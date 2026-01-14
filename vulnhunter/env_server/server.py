"""
VulnHunter OpenEnv Server
Main environment logic for security vulnerability detection and patching.
Implements the OpenEnv Environment interface.
"""
import subprocess
import os
import shutil
import tempfile
from typing import Tuple, Dict, List, Any, Optional

# Try to import from OpenEnv, fallback if not available
try:
    from openenv.core.env_server import Environment, create_fastapi_app
    OPENENV_AVAILABLE = True
except ImportError:
    from abc import ABC, abstractmethod
    OPENENV_AVAILABLE = False
    
    class Environment(ABC):
        """Fallback Environment base class matching OpenEnv interface."""
        def __init__(self, transform=None):
            self.transform = transform
        
        @abstractmethod
        def reset(self, seed=None, episode_id=None, **kwargs):
            pass
        
        @abstractmethod
        def step(self, action, timeout_s=None, **kwargs):
            pass
        
        @property
        @abstractmethod
        def state(self):
            pass

from .models import (
    VulnHunterAction,
    VulnHunterObservation,
    VulnHunterState,
    VulnerabilityType
)


class VulnHunterEnv(Environment[VulnHunterAction, VulnHunterObservation, VulnHunterState]):
    """Security vulnerability training environment.
    
    Implements the OpenEnv Environment interface for RL training.
    Agents learn to identify and patch security vulnerabilities.
    """
    
    MAX_STEPS: int = 15
    VULN_APP_PATH: str = "/app/vulnerable_app"
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    
    def __init__(self, transform=None) -> None:
        super().__init__(transform=transform)
        self._state = VulnHunterState()
        self.workspace: str = tempfile.mkdtemp(prefix="vulnhunter_")
        
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        vuln_type: str = "sql_injection",
        **kwargs: Any
    ) -> VulnHunterObservation:
        """Reset the environment with a new vulnerability scenario.
        
        Args:
            seed: Random seed for reproducibility
            episode_id: Unique episode identifier
            vuln_type: Type of vulnerability to train on
            **kwargs: Additional reset parameters
            
        Returns:
            Initial observation for the new episode
        """
        vt = VulnerabilityType(vuln_type)
        self._state = VulnHunterState(
            current_vuln_type=vt,
            episode_id=episode_id
        )
        
        # Copy vulnerable app to workspace
        if os.path.exists(self.VULN_APP_PATH):
            shutil.copytree(self.VULN_APP_PATH, f"{self.workspace}/app", dirs_exist_ok=True)
        
        obs = VulnHunterObservation(
            stdout="VulnHunter Environment Initialized",
            current_task=f"Find and patch the {vt.value} vulnerability",
            hints=[
                "The vulnerable app is a Flask application",
                "Start by reading the source code with read_file",
                "Use identify_vuln to mark vulnerabilities",
                "Submit a patch to fix the vulnerability"
            ],
            done=False,
            reward=0.0
        )
        
        return self._apply_transform(obs) if self.transform else obs
    
    def step(
        self,
        action: VulnHunterAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any
    ) -> VulnHunterObservation:
        """Execute an action and return the observation.
        
        Args:
            action: The action to execute
            timeout_s: Optional timeout in seconds
            **kwargs: Additional step parameters
            
        Returns:
            Observation containing result, reward, and done flag
        """
        self._state.step_count += 1
        reward: float = 0.0
        obs = VulnHunterObservation()
        
        if action.command:
            obs = self._execute_command(action.command)
        elif action.read_file:
            obs = self._read_source_file(action.read_file)
            reward = 0.05
        elif action.identify_vuln:
            obs, reward = self._identify_vulnerability(action.identify_vuln)
        elif action.patch:
            obs, reward = self._apply_patch(action.patch)
            
        if action.terminate or self._state.step_count >= self.MAX_STEPS:
            self._state.is_terminal = True
            
        self._state.total_reward += reward
        
        # Set reward and done on observation (OpenEnv pattern)
        obs.reward = reward
        obs.done = self._state.is_terminal
        obs.metadata = {
            "step": self._state.step_count,
            "total_reward": self._state.total_reward,
            "vuln_identified": self._state.vuln_identified,
            "patch_successful": self._state.patch_successful
        }
        
        return self._apply_transform(obs) if self.transform else obs
    
    @property
    def state(self) -> VulnHunterState:
        """Get the current environment state."""
        return self._state
    
    def _apply_transform(self, observation: VulnHunterObservation) -> VulnHunterObservation:
        """Apply transform if one is provided."""
        if self.transform is not None:
            return self.transform(observation)
        return observation
    
    def close(self) -> None:
        """Clean up resources."""
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace, ignore_errors=True)
    
    def _execute_command(self, cmd: str) -> VulnHunterObservation:
        """Execute a shell command safely."""
        allowed_prefixes: List[str] = ["curl", "ls", "cat", "grep", "echo", "sqlmap", "nikto"]
        if not any(cmd.strip().startswith(p) for p in allowed_prefixes):
            return VulnHunterObservation(
                stderr=f"Command not allowed. Use: {allowed_prefixes}",
                exit_code=1
            )
        
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=30, cwd=self.workspace
            )
            return VulnHunterObservation(
                stdout=result.stdout[:2000] if result.stdout else "",
                stderr=result.stderr[:500] if result.stderr else "",
                exit_code=result.returncode
            )
        except subprocess.TimeoutExpired:
            return VulnHunterObservation(stderr="Command timed out", exit_code=-1)
        except Exception as e:
            return VulnHunterObservation(stderr=str(e), exit_code=-1)
    
    def _read_source_file(self, filepath: str) -> VulnHunterObservation:
        """Read a source file from the vulnerable app."""
        safe_path = os.path.normpath(os.path.join(self.workspace, filepath))
        if not safe_path.startswith(self.workspace):
            return VulnHunterObservation(stderr="Path traversal detected!", exit_code=1)
        
        try:
            with open(safe_path, "r") as f:
                return VulnHunterObservation(source_code=f.read())
        except FileNotFoundError:
            return VulnHunterObservation(stderr=f"File not found: {filepath}", exit_code=1)
    
    def _identify_vulnerability(self, vuln_info: Dict[str, str]) -> Tuple[VulnHunterObservation, float]:
        """Check if the agent correctly identified the vulnerability."""
        reported_type = vuln_info.get("type", "").lower()
        expected_type = self._state.current_vuln_type.value
        
        if reported_type == expected_type:
            self._state.vuln_identified = True
            return VulnHunterObservation(
                stdout=f"Correct! {expected_type} vulnerability identified."
            ), 0.3
        else:
            return VulnHunterObservation(
                stdout=f"Incorrect. {reported_type} is not the primary vulnerability."
            ), -0.1
    
    def _apply_patch(self, patch: Dict[str, str]) -> Tuple[VulnHunterObservation, float]:
        """Apply a patch and verify it fixes the vulnerability."""
        for filepath, new_content in patch.items():
            safe_path = os.path.normpath(os.path.join(self.workspace, filepath))
            if not safe_path.startswith(self.workspace):
                return VulnHunterObservation(stderr="Invalid patch path"), -0.2
            
            try:
                with open(safe_path, "w") as f:
                    f.write(new_content)
                
                if self._verify_patch_security():
                    self._state.patch_successful = True
                    self._state.is_terminal = True
                    return VulnHunterObservation(stdout="PATCH SUCCESSFUL! Vulnerability fixed."), 1.0
                else:
                    return VulnHunterObservation(stdout="Patch applied but vulnerability still exploitable."), 0.2
                    
            except SyntaxError:
                return VulnHunterObservation(stderr="Patch has syntax errors"), -0.2
            except Exception as e:
                return VulnHunterObservation(stderr=str(e)), -0.1
        
        return VulnHunterObservation(stderr="No valid patches provided"), 0.0
    
    def _verify_patch_security(self) -> bool:
        """Verify that the patch actually fixes the vulnerability."""
        vuln_type = self._state.current_vuln_type
        app_path = f"{self.workspace}/app.py"
        
        if not os.path.exists(app_path):
            return False
            
        with open(app_path, "r") as f:
            code = f.read()
        
        if vuln_type == VulnerabilityType.SQL_INJECTION:
            if "?" in code and "execute" in code:
                return True
            if "%()" in code or "%s" in code:
                return True
            if "f\"SELECT" not in code and "f'SELECT" not in code:
                return True
        
        elif vuln_type == VulnerabilityType.XSS:
            if "escape(" in code or "Markup(" in code:
                return True
        
        elif vuln_type == VulnerabilityType.PATH_TRAVERSAL:
            if "safe_join(" in code or "secure_filename(" in code:
                return True
        
        return False


# Create FastAPI Application using OpenEnv helper if available
env = VulnHunterEnv()

if OPENENV_AVAILABLE:
    # Use OpenEnv's create_fastapi_app for full compatibility
    app = create_fastapi_app(env)
else:
    # Fallback: Create FastAPI app manually
    from fastapi import FastAPI
    
    app = FastAPI(
        title="VulnHunter Environment",
        version="1.0.0",
        description="OpenEnv-compatible security vulnerability training environment"
    )
    
    @app.post("/reset")
    async def reset_endpoint(vuln_type: str = "sql_injection") -> dict:
        obs = env.reset(vuln_type=vuln_type)
        return obs.model_dump()
    
    @app.post("/step")
    async def step_endpoint(action: VulnHunterAction) -> dict:
        obs = env.step(action)
        return obs.model_dump()
    
    @app.get("/state")
    async def state_endpoint() -> dict:
        return env.state.model_dump()
    
    @app.get("/health")
    async def health_endpoint() -> dict:
        return {"status": "ok"}
    
    @app.get("/schema")
    async def schema_endpoint() -> dict:
        return {
            "action": VulnHunterAction.model_json_schema(),
            "observation": VulnHunterObservation.model_json_schema(),
            "state": VulnHunterState.model_json_schema()
        }
