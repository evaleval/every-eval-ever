"""
Centralized project configuration and registries.

Edit this file to update environment-specific paths, dataset/model registries,
and frequently changed constants. This module is intentionally minimal and free
of side-effects so it can be safely imported anywhere.

Guidelines:
- Keep values deterministic and static; do not compute values from runtime state.
- If you change paths, ensure they exist or are created by call sites (not here).
- If you add dataset/model entries, follow existing key naming conventions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

# --------------------------------------------------------------------------------------
# Paths (environment-specific)
# --------------------------------------------------------------------------------------

# NOTE: Updated paths to work with new project structure
HF_MAP_DATA_DIR: Path = Path(__file__).parent.parent / "data" / "hf_map_data"

# Path to model metadata CSV file used by model utilities  
MODEL_METADATA_CSV: Path = Path(__file__).parent / "model_metadata.csv"

# Base directory for all data files (now in project root)
DATA_DIR: Path = Path(__file__).parent.parent / "data"

# Directory for storing downloaded benchmark CSVs.
BENCHMARK_CSVS_DIR: Path = DATA_DIR / "benchmark_lines"

# Directory paths
DOWNLOADS_DIR: Path = DATA_DIR / "downloads"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
AGGREGATED_DATA_DIR: Path = DATA_DIR / "aggregated"

# --------------------------------------------------------------------------------------
# HELM download & processing settings
# --------------------------------------------------------------------------------------

# Versions to search for when downloading HELM files (kept identical ordering)
HELM_1_VERSIONS: List[str] = [f"v1.{i}.0" for i in range(14)]  # v1.0.0 to v1.13.0
HELM_0_VERSIONS: List[str] = [f"v0.{i}.0" for i in range(3, 14)]  # v1.0.0 to v1.13.0
HELM_VERSIONS: List[str] = HELM_1_VERSIONS + HELM_0_VERSIONS
# Default starting version
DEFAULT_START_VERSION: str = "v1.0.0"

# Base URL template for HELM assets
HELM_URL_WITH_BENCHMARK_TEMPLATE: str = (
    "https://storage.googleapis.com/crfm-helm-public/{benchmark}/benchmark_output/runs/{version}"
)

HELM_URL_WITHOUT_BENCHMARK_TEMPLATE: str = (
    "https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/{version}"
)

# File types fetched per task (unchanged)
HELM_FILE_TYPES: List[str] = [
    "run_spec",
    "stats",
    "per_instance_stats",
    "instances",
    "scenario_state",
    "display_predictions",
    "display_requests",
    "scenario",
]

# Concurrency for `ProcessPoolExecutor` in `helm_data_processor.py`
# Set to 1 by default to avoid log message conflicts and overlapping progress bars
# Increase for faster processing if clean logs aren't needed
PROCESS_POOL_MAX_WORKERS: int = 1

# --------------------------------------------------------------------------------------
# Small utility constants
# --------------------------------------------------------------------------------------

# Progress bar format used in `helm_data_processor.py` (kept identical)
TQDM_BAR_FORMAT: str = (
    "{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)
