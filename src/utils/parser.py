import yaml
import os
from pathlib import Path


def get_config(config_path: str) -> dict:
    """
    Get the config.

    Args:
        config_path (str): The path to the config file

    Returns:
        dict: The config
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config file not found: {Path(config_path).resolve()}")
    
    config = yaml.safe_load(Path(config_path).read_text())
    
    return config