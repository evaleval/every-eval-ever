import json
import logging
import os
from typing import List

import colorama
import requests
from colorama import Fore, Style
from tqdm import tqdm

# Import centralized settings (paths, constants)
from config.settings import (
    HELM_VERSIONS,
    HELM_FILE_TYPES,
    HELM_URL_WITH_BENCHMARK_TEMPLATE,
    HELM_URL_WITHOUT_BENCHMARK_TEMPLATE,
    DOWNLOADS_DIR,
    PROCESSED_DATA_DIR,
    DEFAULT_START_VERSION,
    TQDM_BAR_FORMAT,
)

# Import centralized logging
from src.utils.logging_config import setup_logging

# Initialize colorama
colorama.init()

# Set up logging using centralized configuration
logger = setup_logging("helm_processor.log", "HELM_Downloader")


# Custom logger function with emojis and colors
def log_info(message, emoji="ℹ️"):
    logger.info(f"{emoji} {message}")
    print(f"{Fore.CYAN}{emoji} {message}{Style.RESET_ALL}")


def log_success(message, emoji="✅"):
    logger.info(f"{emoji} {message}")
    print(f"{Fore.GREEN}{emoji} {message}{Style.RESET_ALL}")


def log_error(message, emoji="❌"):
    logger.error(f"{emoji} {message}")
    print(f"{Fore.RED}{emoji} {message}{Style.RESET_ALL}")


def log_warning(message, emoji="⚠️"):
    logger.warning(f"{emoji} {message}")
    print(f"{Fore.YELLOW}{emoji} {message}{Style.RESET_ALL}")


def log_step(step_name, emoji="🔄"):
    logger.info(f"{emoji} {step_name}")
    print(f"{Fore.MAGENTA}{emoji} {step_name}{Style.RESET_ALL}")


# Create necessary directories
os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


def get_json_from_url(url):
    """Fetch JSON data from given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        # log_error(f"An error occurred with URL {url}: {e}")
        return None


def _try_get_json_data(task: str, file_type: str, version: str, benchmark: str) -> dict or None:
    """Try to fetch JSON data from both URL patterns."""
    # Try URL with benchmark first
    url_with_benchmark = f"{HELM_URL_WITH_BENCHMARK_TEMPLATE.format(benchmark=benchmark, version=version)}/{task}/{file_type}.json"
    json_data = get_json_from_url(url_with_benchmark)

    # If not found, try URL without benchmark
    if json_data is None:
        url_without_benchmark = f"{HELM_URL_WITHOUT_BENCHMARK_TEMPLATE.format(version=version)}/{task}/{file_type}.json"
        json_data = get_json_from_url(url_without_benchmark)

    return json_data


def _download_file_to_path(json_data: dict, save_path: str) -> None:
    """Save JSON data to file."""
    with open(save_path, "w") as f:
        json.dump(json_data, f, indent=2)


def _setup_task_directory(output_dir: str, task: str, overwrite: bool) -> str:
    """Create and setup task directory with proper permissions."""
    save_dir = os.path.join(output_dir, task)

    # Clean existing directory if overwriting
    if os.path.exists(save_dir) and overwrite:
        try:
            import shutil
            shutil.rmtree(save_dir)
            log_info(f"Removed existing directory: {save_dir}", "🗑️")
        except Exception as e:
            log_warning(f"Could not remove existing directory: {e}", "⚠️")

    # Create directory with full permissions
    os.makedirs(save_dir, exist_ok=True)
    try:
        os.chmod(save_dir, 0o777)
    except Exception as e:
        log_warning(f"Could not set directory permissions: {e}", "⚠️")

    return save_dir


def download_task(task: str, output_dir: str, benchmark: str, overwrite: bool = False,
                  start_version: str = DEFAULT_START_VERSION) -> str or None:
    """
    Download and extract files for a given task.
    This function tries different versions of HELM data.
    """
    log_step(f"Downloading task: {task}", "🔽")

    # Get versions to try (starting from specified version)
    versions = list(HELM_VERSIONS)
    start_idx = versions.index(start_version)
    versions = versions[start_idx:]

    # Setup directory
    save_dir = _setup_task_directory(output_dir, task, overwrite)

    # Initialize statistics
    file_types = list(HELM_FILE_TYPES)
    task_stats = {
        "task": task,
        "total_files": len(file_types),
        "found_files": 0,
        "missing_files": 0,
        "version_usage": {v: 0 for v in versions}
    }

    found_version = None

    for file_type in file_types:
        save_path = os.path.join(save_dir, f"{file_type}.json")

        # Skip existing files if not overwriting
        if os.path.exists(save_path) and not overwrite:
            log_info(f"File {file_type}.json already exists for task {task} (skipping)", "📄")
            task_stats["found_files"] += 1
            continue

        # Determine which version to use
        search_versions = [found_version] if found_version else versions

        # Try to download the file
        downloaded = False
        for version in search_versions:
            json_data = _try_get_json_data(task, file_type, version, benchmark)
            if json_data:
                _download_file_to_path(json_data, save_path)
                task_stats["version_usage"][version] += 1
                task_stats["found_files"] += 1
                downloaded = True
                log_success(f"Found {task}/{file_type}.json in version {version}", "📥")

                # Lock in this version for subsequent files
                if not found_version:
                    found_version = version
                break

        if not downloaded:
            task_stats["missing_files"] += 1
            log_warning(f"Could not find {task}/{file_type}.json in any version", "🔍")

    # Log final results
    success_rate = task_stats["found_files"] / task_stats["total_files"]
    if success_rate == 1.0:
        log_success(f"Downloaded all {task_stats['total_files']} files for task {task}", "🎉")
    else:
        log_warning(
            f"Downloaded {task_stats['found_files']}/{task_stats['total_files']} files for task {task}",
            "⚠️"
        )

    return save_dir


def download_tasks(tasks: List[str], output_dir: str, benchmark: str, overwrite: bool = False,
                   start_version: str = DEFAULT_START_VERSION, show_progress: bool = True) -> List[str]:
    """
    Download a list of tasks and return the paths to the saved files.
    
    Args:
        tasks: List of task names to download
        output_dir: Directory to save downloaded files
        benchmark: Benchmark name
        overwrite: Whether to overwrite existing files
        start_version: HELM version to start from
        show_progress: Whether to show progress bar (disable for cleaner output in sequential mode)
    """
    log_step(f"Downloading {len(tasks)} tasks", "🔽")

    saved_files = []
    task_iterator = tqdm(tasks, desc="Processing tasks") if show_progress else tasks
    
    for task in task_iterator:
        log_step(f"Starting download for task: {task}")
        saved_file = download_task(task, output_dir, benchmark, overwrite, start_version)
        if saved_file:
            saved_files.append(saved_file)
            log_success(f"Successfully downloaded task '{task}' to '{saved_file}'")
        else:
            log_error(f"Failed to download task: {task}")

    return saved_files

