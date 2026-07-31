"""
Download the Elliptic Bitcoin dataset from Hugging Face when local files are missing.

Used by the Streamlit dashboard on Streamlit Cloud and by local setups that
prefer not to copy CSVs manually from Kaggle.
"""

from __future__ import annotations

from pathlib import Path

DATASET_FILES = {
    "raw_classes": "elliptic_txs_classes.csv",
    "raw_edgelist": "elliptic_txs_edgelist.csv",
    "raw_features": "elliptic_txs_features.csv",
}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_paths(config: dict, root: Path | None = None) -> dict:
    """Convert relative path entries in config['paths'] to absolute paths."""
    root = root or get_project_root()
    for key, value in config.get("paths", {}).items():
        path = Path(value)
        if not path.is_absolute():
            config["paths"][key] = str(root / value)
    return config


def dataset_files_present(config: dict) -> bool:
    return all(Path(config["paths"][key]).exists() for key in DATASET_FILES)


def ensure_dataset(config: dict, root: Path | None = None) -> bool:
    """
    Download missing CSV files from Hugging Face.

    Returns True when all three files exist locally (already present or downloaded).
    Returns False when files are missing and no Hugging Face repo is configured.
    """
    root = root or get_project_root()
    resolve_paths(config, root)

    if dataset_files_present(config):
        return True

    hf_cfg = config.get("huggingface") or {}
    repo_id = hf_cfg.get("repo_id")
    if not repo_id:
        return False

    from huggingface_hub import hf_hub_download

    raw_dir = Path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    for path_key, filename in DATASET_FILES.items():
        dest = Path(config["paths"][path_key])
        if dest.exists():
            continue
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(raw_dir),
        )

    return dataset_files_present(config)
