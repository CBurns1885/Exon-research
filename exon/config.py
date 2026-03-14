"""Configuration loading from YAML files and environment variables."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def load_config(path: str | Path | None = None) -> dict:
    """Load configuration, merging defaults with local overrides.

    Priority: environment variables > local.yaml > default.yaml
    """
    base_dir = Path(__file__).parent.parent / "config"

    # Load default
    default_path = base_dir / "default.yaml"
    config = {}
    if default_path.exists():
        with open(default_path) as f:
            config = yaml.safe_load(f) or {}

    # Load local override
    if path:
        override_path = Path(path)
    else:
        override_path = base_dir / "local.yaml"

    if override_path.exists():
        with open(override_path) as f:
            local = yaml.safe_load(f) or {}
        config = _deep_merge(config, local)

    # Environment variable overrides — Coinbase
    env_key = os.environ.get("COINBASE_API_KEY")
    env_secret = os.environ.get("COINBASE_API_SECRET")
    if env_key:
        config.setdefault("coinbase", {})["api_key"] = env_key
    if env_secret:
        config.setdefault("coinbase", {})["api_secret"] = env_secret

    # Environment variable overrides — Alpaca
    alpaca_key = os.environ.get("ALPACA_API_KEY")
    alpaca_secret = os.environ.get("ALPACA_API_SECRET")
    if alpaca_key:
        config.setdefault("alpaca", {})["api_key"] = alpaca_key
    if alpaca_secret:
        config.setdefault("alpaca", {})["api_secret"] = alpaca_secret

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
