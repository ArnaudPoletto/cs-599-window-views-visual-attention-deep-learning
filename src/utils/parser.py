import yaml
from pathlib import Path


def get_config(config_path: str) -> dict:
    """
    Get the config.

    Args:
        config_path (str): The path to the config file

    Returns:
        dict: The config
    """
    config = yaml.safe_load(Path(config_path).read_text())
    return config