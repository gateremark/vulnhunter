"""
VulnHunter OpenEnv Server
Main environment logic for security vulnerability detection and patching.
"""
import subprocess
import os
import shutil
import tempfile
from typing import Tuple, Dict, List
from fastapi import FastAPI
from .models import Action, Observation, State, StepResult, VulnerabilityType


class VulnHunterEnv:
    """Security vulnerability training environment."""
    
    MAX_STEPS: int = 15
    VULN_APP_PATH: str = "/app/vulnerable_app"
    
    def __init__(self) -> None:
        self.state = State()
        self.workspace: str = tempfile.mkdtemp(prefix="vulnhunter_")
        
    def reset(self, vuln_type: VulnerabilityType = VulnerabilityType.SQL_INJECTION) -> Observation:
        """Reset the environment with a new vulnerability scenario."""
        self.state = State(current_vuln_type=vuln_type)
        
        if os.path.exists(self.VULN_APP_PATH):
            shutil.copytree(self.VULN_APP_PATH, f"{self.workspace}/app", dirs_exist_ok=True)
        
        return Observation(
            stdout=str("VulnHunter Environment Initialized"),
            current_task=str(f"Find and patch the {vuln_type.value} vulnerability"),
            hints=[
                "The vulnerable app is a Flask application",
                "Start by reading the source code with read_file",
                "Use identify_vuln to mark vulnerabilities",
                "Submit a patch to fix the vulnerability"
            ]
        )
    
    def step(self, action: Action) -> StepResult:
        """Execute an action and return the result."""
        self.state.step_count += 1
        reward: float = 0.0
        obs = Observation()
        
        if action.command:
            obs = self._execute_command(action.command)
        elif action.read_file:
            obs = self._read_source_file(action.read_file)
            reward = 0.05
        elif action.identify_vuln:
            obs, reward = self._identify_vulnerability(action.identify_vuln)
        elif action.patch:
            obs, reward = self._apply_patch(action.patch)
            
        if action.terminate or self.state.step_count >= self.MAX_STEPS:
            self.state.is_terminal = True
            
        self.state.total_reward += reward
        
        return StepResult(
            observation=obs,
            reward=reward,
            done=self.state.is_terminal,
            info={
                "step": self.state.step_count,
                "total_reward": self.state.total_reward,
                "vuln_identified": self.state.vuln_identified,
                "patch_successful": self.state.patch_successful
            }
        )
    
    def _execute_command(self, cmd: str) -> Observation:
        """Execute a shell command safely."""
        allowed_prefixes: List[str] = ["curl", "ls", "cat", "grep", "echo", "sqlmap", "nikto"]
        if not any(cmd.strip().startswith(p) for p in allowed_prefixes):
            return Observation(
                stderr=str(f"Command not allowed. Use: {allowed_prefixes}"),
                exit_code=1
            )
        
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=30, cwd=self.workspace
            )
            return Observation(
                stdout=str(result.stdout[:2000]) if result.stdout else "",
                stderr=str(result.stderr[:500]) if result.stderr else "",
                exit_code=result.returncode
            )
        except subprocess.TimeoutExpired:
            return Observation(stderr="Command timed out", exit_code=-1)
        except Exception as e:
            return Observation(stderr=str(e), exit_code=-1)
    
    def _read_source_file(self, filepath: str) -> Observation:
        """Read a source file from the vulnerable app."""
        safe_path = os.path.normpath(os.path.join(self.workspace, filepath))
        if not safe_path.startswith(self.workspace):
            return Observation(stderr="Path traversal detected!", exit_code=1)
        
        try:
            with open(safe_path, "r") as f:
                return Observation(source_code=f.read())
        except FileNotFoundError:
            return Observation(stderr=f"File not found: {filepath}", exit_code=1)
    
    def _identify_vulnerability(self, vuln_info: Dict[str, str]) -> Tuple[Observation, float]:
        """Check if the agent correctly identified the vulnerability."""
        reported_type = vuln_info.get("type", "").lower()
        expected_type = self.state.current_vuln_type.value
        
        if reported_type == expected_type:
            self.state.vuln_identified = True
            return Observation(
                stdout=f"Correct! {expected_type} vulnerability identified."
            ), 0.3
        else:
            return Observation(
                stdout=f"Incorrect. {reported_type} is not the primary vulnerability."
            ), -0.1
    
    def _apply_patch(self, patch: Dict[str, str]) -> Tuple[Observation, float]:
        """Apply a patch and verify it fixes the vulnerability."""
        for filepath, new_content in patch.items():
            safe_path = os.path.normpath(os.path.join(self.workspace, filepath))
            if not safe_path.startswith(self.workspace):
                return Observation(stderr="Invalid patch path"), -0.2
            
            try:
                with open(safe_path, "w") as f:
                    f.write(new_content)
                
                if self._verify_patch_security():
                    self.state.patch_successful = True
                    self.state.is_terminal = True
                    return Observation(stdout="PATCH SUCCESSFUL! Vulnerability fixed."), 1.0
                else:
                    return Observation(stdout="Patch applied but vulnerability still exploitable."), 0.2
                    
            except SyntaxError:
                return Observation(stderr="Patch has syntax errors"), -0.2
            except Exception as e:
                return Observation(stderr=str(e)), -0.1
        
        return Observation(stderr="No valid patches provided"), 0.0
    
    def _verify_patch_security(self) -> bool:
        """Verify that the patch actually fixes the vulnerability."""
        vuln_type = self.state.current_vuln_type
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
        
        return False


# FastAPI Application
app = FastAPI(title="VulnHunter Environment", version="1.0.0")
env = VulnHunterEnv()


@app.post("/reset")
async def reset_endpoint(vuln_type: str = "sql_injection") -> Observation:
    vt = VulnerabilityType(vuln_type)
    return env.reset(vt)


@app.post("/step")
async def step_endpoint(action: Action) -> StepResult:
    return env.step(action)


@app.get("/state")
async def state_endpoint() -> State:
    return env.state


@app.get("/health")
async def health_endpoint() -> dict:
    return {"status": "ok"}
