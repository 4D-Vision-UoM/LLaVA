import os
import yaml
from pathlib import Path

class ConfigManager:
    """Handles path resolution and API key loading."""
    def __init__(self):
        self.current_file_path = Path(__file__).resolve()
        # Assumes this file is inside tools/ or src/, so parent.parent is root
        self.project_root = self.current_file_path.parent.parent
        self.config_path = self.project_root / "config" / "openai_config.yaml"
        
    def load_api_key(self):
        if not self.config_path.exists():
            fallback_path = Path("configs/openrouter/OpenRouter_API.yaml")
            if fallback_path.exists():
                self.config_path = fallback_path
            else:
                raise FileNotFoundError(f"Config file not found at {self.config_path}")

        print(f"Loading config from {self.config_path}")
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
            api_key = config.get("OPENROUTER_API_KEY") or config.get("openrouter_api_key")
            if not api_key:
                raise ValueError(f"OPENROUTER_API_KEY not found in config file {self.config_path}")
            os.environ["OPENROUTER_API_KEY"] = api_key
            return api_key