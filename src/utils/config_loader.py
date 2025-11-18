"""Configuration loading utilities"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent.parent


def load_config() -> Dict[str, Any]:
    """Load main configuration from YAML file"""
    config_path = get_project_root() / "config" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_keywords() -> Dict[str, Any]:
    """Load keywords configuration from YAML file"""
    keywords_path = get_project_root() / "config" / "keywords.yaml"

    if not keywords_path.exists():
        raise FileNotFoundError(f"Keywords file not found: {keywords_path}")

    with open(keywords_path, 'r') as f:
        keywords = yaml.safe_load(f)

    return keywords


def load_env():
    """Load environment variables from .env file"""
    from dotenv import load_dotenv

    env_path = get_project_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        print("Warning: .env file not found. Using environment variables.")


def get_api_key(service: str) -> str:
    """Get API key for a specific service from environment variables"""
    key_map = {
        'twitter': 'TWITTER_BEARER_TOKEN',
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'fred': 'FRED_API_KEY'
    }

    key_name = key_map.get(service.lower())
    if not key_name:
        raise ValueError(f"Unknown service: {service}")

    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(f"API key not found for {service}. Set {key_name} in .env file")

    return api_key
