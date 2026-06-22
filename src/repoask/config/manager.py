from __future__ import annotations

import sys
from pathlib import Path

import tomli_w
from pydantic import ValidationError

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from .schema import RepoAskConfig

GLOBAL_CONFIG_DIR = Path.home() / ".repoask"
GLOBAL_CONFIG_FILE = GLOBAL_CONFIG_DIR / "config.toml"


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(repo_root: Path | None = None) -> RepoAskConfig:
    data: dict = {}

    if GLOBAL_CONFIG_FILE.exists():
        with open(GLOBAL_CONFIG_FILE, "rb") as f:
            data = tomllib.load(f)

    if repo_root is not None:
        local_file = repo_root / ".repoask" / "config.toml"
        if local_file.exists():
            with open(local_file, "rb") as f:
                local_data = tomllib.load(f)
            data = _deep_merge(data, local_data)

    try:
        return RepoAskConfig.model_validate(data)
    except ValidationError as e:
        raise SystemExit(f"[repoask] Invalid config: {e}") from e


def save_global_config(config: RepoAskConfig) -> None:
    GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_CONFIG_FILE, "wb") as f:
        tomli_w.dump(config.model_dump(), f)


def get_or_create_global_config() -> RepoAskConfig:
    if not GLOBAL_CONFIG_FILE.exists():
        config = RepoAskConfig()
        save_global_config(config)
        return config
    return load_config()
