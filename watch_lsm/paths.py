"""Project-relative defaults with explicit, ignored per-machine overrides."""

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def machine_config():
    path = PROJECT_ROOT / "watch_lsm.local.toml"
    if not path.is_file():
        return {}
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with path.open("rb") as stream:
        return tomllib.load(stream)


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def data_path(*parts):
    value = os.environ.get("WATCH_LSM_DATASETS_DIR") or machine_config().get("paths", {}).get("datasets", "datasets")
    return project_path(value).joinpath(*parts)


def model_path(*parts):
    value = os.environ.get("WATCH_LSM_MODELS_DIR") or machine_config().get("paths", {}).get("models", "models")
    return project_path(value).joinpath(*parts)


def result_path(*parts):
    value = os.environ.get("WATCH_LSM_RESULTS_DIR") or machine_config().get("paths", {}).get("results", "results")
    return project_path(value).joinpath(*parts)


def pulsedb_source():
    value = os.environ.get("WATCH_LSM_PULSEDB_SOURCE") or machine_config().get("paths", {}).get("pulsedb_source")
    if not value:
        raise ValueError("Set --source, WATCH_LSM_PULSEDB_SOURCE or paths.pulsedb_source in watch_lsm.local.toml")
    return project_path(value)


def load_yaml(path):
    with project_path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)
