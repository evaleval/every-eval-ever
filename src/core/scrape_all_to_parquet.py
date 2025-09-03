"""
Orchestrate scraping of HELM benchmarks, download/convert all tasks, and
aggregate all processed CSVs into a single Parquet file using a detailed
pyarrow schema.

Usage (from repo root):
    python -m src.core.scrape_all_to_parquet --output /path/to/output.parquet

This script re-uses existing modules in the repository:
- src.core.web_scraper (to create benchmark CSVs)
- main_processor.main (to download+convert tasks listed in CSVs)

Notes:
- Network access required to scrape and download HELM assets.
- By default this will process the example benchmarks `lite`, `mmlu`,
  and `classic` as well as any `helm_*.csv` files already present under
  the configured `BENCHMARK_CSVS_DIR`.
"""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
from typing import List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timezone

from config.settings import (
    BENCHMARK_CSVS_DIR,
    PROCESSED_DATA_DIR,
    AGGREGATED_DATA_DIR,
)

# Reuse existing entry points
import subprocess
import sys
from shutil import which


# Default benchmark candidates (these are common examples in the README)
DEFAULT_BENCHMARKS = ["lite", "mmlu", "classic"]


def discover_benchmarks() -> List[str]:
    """Discover benchmarks to process.

    Strategy:
    - Look for existing CSVs in BENCHMARK_CSVS_DIR (files named helm_{benchmark}.csv)
    - Merge with DEFAULT_BENCHMARKS
    """
    BENCHMARK_CSVS_DIR.mkdir(parents=True, exist_ok=True)
    discovered = set()
    for p in BENCHMARK_CSVS_DIR.glob("helm_*.csv"):
        name = p.stem.replace("helm_", "")
        if name:
            discovered.add(name)

    discovered.update(DEFAULT_BENCHMARKS)
    return sorted(discovered)


def ensure_csv_for_benchmark(benchmark: str) -> Path:
    """Ensure a `helm_{benchmark}.csv` exists; create it by scraping if missing.

    Returns the path to the CSV file.
    """
    csv_path = BENCHMARK_CSVS_DIR / f"helm_{benchmark}.csv"
    if not csv_path.exists():
        print(f"CSV for benchmark '{benchmark}' not found, scraping HELM website (external process)...")
        python_exec = sys.executable
        # Try to run the web_scraper module as a script: python -m src.core.web_scraper
        cmd = [python_exec, "-m", "src.core.web_scraper", "--benchmark", benchmark, "--output-dir", str(BENCHMARK_CSVS_DIR)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Failed to run web_scraper. Return code: {res.returncode}\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
            raise RuntimeError(f"web_scraper failed for benchmark '{benchmark}'")
        if not csv_path.exists():
            raise FileNotFoundError(f"Scraper finished but CSV not created for '{benchmark}'")
        print(f"Created CSV: {csv_path}")
    else:
        print(f"Found existing CSV: {csv_path}")
    return csv_path


def process_benchmark_csv(csv_path: Path, benchmark: str, downloads_dir: Path | None = None,
                          keep_temp: bool = False, overwrite: bool = False, max_workers: int = 8):
    """Call the project's `main_processor.main` to download & convert all tasks in a CSV.

    `main_processor.main` expects: csv_file (str), output_dir (Path), benchmark (str), ...
    We reuse it to avoid duplicating download/convert logic.
    """
    output_dir = PROCESSED_DATA_DIR / benchmark
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the main_processor.py script as a subprocess to avoid import-time side-effects
    python_exec = sys.executable
    cmd = [
        python_exec,
        str(Path.cwd() / "main_processor.py"),
        "--benchmark",
        benchmark,
        "--downloads-dir",
        str(downloads_dir) if downloads_dir else str(Path(PROCESSED_DATA_DIR).parent / "downloads"),
    ]
    if keep_temp:
        cmd.append("--keep-temp")
    if overwrite:
        cmd.append("--overwrite")
    if max_workers:
        cmd.extend(["--max-workers", str(max_workers)])

    print(f"Running processor for benchmark '{benchmark}' via: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Processor failed for benchmark '{benchmark}' (rc={res.returncode})\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
        raise RuntimeError(f"Processor failed for benchmark '{benchmark}'")
    print(f"Processor completed for benchmark '{benchmark}'")


def _build_pyarrow_schema() -> pa.Schema:
    """Return a detailed pyarrow schema for the final Parquet file.

    The schema is intentionally wider than the CSVs produced by the converter so
    that additional metadata can be stored. Fields are nullable by default.
    """
    # Generic, extendable schema to support multiple evaluation sources (not only HELM)
    return pa.schema([
        ("evaluation_id", pa.string()),
        ("dataset_name", pa.string()),
        ("hf_split", pa.string()),
        ("hf_index", pa.int64()),
        ("raw_input", pa.string()),
        ("ground_truth", pa.string()),
        ("model_name", pa.string()),
        ("model_family", pa.string()),
        ("output", pa.string()),
        ("evaluation_method_name", pa.string()),
        ("evaluation_score", pa.float64()),

        # Provenance / metadata
        ("run", pa.string()),
        ("task", pa.string()),
        ("adapter_method", pa.string()),
        ("source", pa.string()),                # e.g., 'helm', 'other_source'
        ("source_version", pa.string()),       # version or commit of the source
        ("source_url", pa.string()),
        ("ingestion_timestamp", pa.timestamp('ns')),  # when row was ingested into the parquet
        ("license", pa.string()),
        ("category", pa.string()),              # risk/capability category (one of 20); left blank for now
        ("source_csv", pa.string()),
        ("source_benchmark", pa.string()),
    ])


def aggregate_all_processed_to_parquet(output_parquet: Path, input_dir: Path = PROCESSED_DATA_DIR,
                                      schema: pa.Schema | None = None, batch_size: int = 20):
    """Aggregate all CSV files under `input_dir/*/*.csv` into a single Parquet file.

    - Ensures the resulting parquet conforms to `schema` (adds missing columns as null)
    - Writes a single file at `output_parquet`
    """
    if schema is None:
        schema = _build_pyarrow_schema()

    csv_paths = sorted([p for p in input_dir.rglob("*.csv") if p.is_file()])
    if not csv_paths:
        raise FileNotFoundError(f"No processed CSV files found under {input_dir}")

    out_dir = output_parquet.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_parquet.exists():
        output_parquet.unlink()

    writer: pq.ParquetWriter | None = None
    total_rows = 0

    for i in range(0, len(csv_paths), batch_size):
        batch = csv_paths[i: i + batch_size]
        frames = []
        for csv_p in batch:
            try:
                df = pd.read_csv(csv_p)
            except Exception as e:
                print(f"[skip] Failed reading {csv_p}: {e}")
                continue

            # Record provenance
            df["source_csv"] = str(csv_p.name)
            # Assume parent folder name is the benchmark
            df["source_benchmark"] = str(csv_p.parent.name)

            # Set a generic `source` field so this Parquet can include non-HELM data in future.
            # Current processed CSVs are produced by the HELM pipeline, so mark as 'helm'.
            df["source"] = "helm"

            # source_version is not always known; leave null for now
            df["source_version"] = pd.NA
            df["source_url"] = pd.NA
            # ingestion timestamp (UTC)
            df["ingestion_timestamp"] = pd.Timestamp(datetime.now(timezone.utc))
            df["license"] = pd.NA
            # category is reserved for one of 20 risk/capability categories — leave blank for now
            df["category"] = pd.NA

            # Ensure columns in schema exist (use schema names list)
            for name in schema.names:
                if name not in df.columns:
                    df[name] = pd.NA

            # Reorder columns to schema order
            df = df[[name for name in schema.names]]

            frames.append(df)

        if not frames:
            continue

        batch_df = pd.concat(frames, ignore_index=True)

        # Cast to pyarrow table using provided schema (preserve nullability)
        table = pa.Table.from_pandas(batch_df, schema=schema, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(output_parquet, table.schema)
        writer.write_table(table)
        total_rows += table.num_rows
        print(f"Wrote batch of {table.num_rows:,} rows (total {total_rows:,})")

    if writer is not None:
        writer.close()
    else:
        raise RuntimeError("No data written to parquet. Check processed CSV files.")

    print(f"Done. Aggregated {total_rows:,} rows -> {output_parquet}")
    return output_parquet


def main(output: str, benchmarks: List[str] | None = None, downloads_dir: str | None = None,
         keep_temp: bool = False, overwrite: bool = False, max_workers: int = 8):
    """High level orchestration.

    Steps:
    1. Discover benchmarks
    2. Ensure CSVs exist (scrape if necessary)
    3. For each benchmark: download+convert all tasks to processed CSVs
    4. Aggregate all processed CSVs into a single Parquet file with schema
    """
    output_path = Path(output)
    downloads_path = Path(downloads_dir) if downloads_dir else None

    benchmarks_to_run = benchmarks if benchmarks else discover_benchmarks()
    print(f"Benchmarks to process: {benchmarks_to_run}")

    for b in benchmarks_to_run:
        try:
            csv_path = ensure_csv_for_benchmark(b)
            print(f"Processing benchmark '{b}' (CSV: {csv_path})")
            process_benchmark_csv(csv_path, b, downloads_dir=downloads_path,
                                  keep_temp=keep_temp, overwrite=overwrite, max_workers=max_workers)
        except Exception as e:
            print(f"[error] Benchmark '{b}' failed: {e}")

    # Aggregate
    aggregate_all_processed_to_parquet(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape all HELM data and aggregate to a single Parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file path")
    parser.add_argument("--benchmarks", nargs="*", help="Optional list of benchmarks to process (overrides discovery)")
    parser.add_argument("--downloads-dir", help="Optional downloads directory to use for raw HELM JSON files")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary downloaded files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs / re-download")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel workers for processing")
    args = parser.parse_args()

    main(output=args.output, benchmarks=args.benchmarks, downloads_dir=args.downloads_dir,
         keep_temp=args.keep_temp, overwrite=args.overwrite, max_workers=args.max_workers)
