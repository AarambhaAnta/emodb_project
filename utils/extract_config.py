"""
Configuration management utilities for EmoDb project.
"""
import os
import yaml
from pathlib import Path


def get_project_root():
    """
    Get the project root directory.
    
    Returns:
        Path: Absolute path to project root directory
    """
    # Get the directory containing this file (utils/)
    current_file = Path(__file__).resolve()
    # Go up two levels: utils/ -> project_root/
    project_root = current_file.parent.parent
    return project_root


def get_default_config_path():
    """
    Get the default config file path.
    
    Returns:
        str: Absolute path to emodb_config.yaml
    """
    project_root = get_project_root()
    config_path = os.path.join(project_root, 'config', 'emodb_config.yaml')
    return config_path


def get_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str, optional): Path to config file. 
            If None, uses default config/emodb_config.yaml
    
    Returns:
        dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if config_path is None:
        config_path = get_default_config_path()
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as config_file:
        config_data = yaml.safe_load(config_file)
    
    return config_data


def get_path(config, *path_parts):
    """
    Get absolute path by joining config BASE_DIR with path parts.
    
    Args:
        config (dict): Configuration dictionary
        *path_parts: Path components to join
    
    Returns:
        str: Absolute path
        
    Example:
        >>> config = get_config()
        >>> get_path(config, 'data', 'raw', 'emodb')
        '/path/to/project/data/raw/emodb'
    """
    base_dir = config.get('BASE_DIR', get_project_root())
    return os.path.join(base_dir, *path_parts)