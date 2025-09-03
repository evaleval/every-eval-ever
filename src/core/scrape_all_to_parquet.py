"""
Orchestrate scraping of HELM benchmarks, download/convert all tasks, and
aggregate all processed CSVs into a single Parquet file using a detailed
pyarrow schema.

Usage (from repo root):
    python -m src.core.scrape_all_to_parquet --output /path/to/output.parquet

This script re-uses existing modules in the repository:
- src.core.web_scraper (to create benchmark CSVs)
- main.py (unified entry point to download+convert tasks listed in CSVs)

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
import logging
import sys
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
from shutil import which

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('scraper.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


# Default benchmark candidates (these are common examples in the README)
DEFAULT_BENCHMARKS = ["lite", "mmlu", "classic"]


def discover_benchmarks() -> List[str]:
    """Discover benchmarks to process.

    Strategy:
    - Look for existing CSVs in BENCHMARK_CSVS_DIR (files named helm_{benchmark}.csv)
    - Merge with DEFAULT_BENCHMARKS
    """
    logger.info("Discovering benchmarks to process...")
    
    BENCHMARK_CSVS_DIR.mkdir(parents=True, exist_ok=True)
    discovered = set()
    
    existing_files = list(BENCHMARK_CSVS_DIR.glob("helm_*.csv"))
    logger.info(f"Found {len(existing_files)} existing benchmark CSV files in {BENCHMARK_CSVS_DIR}")
    
    for p in existing_files:
        name = p.stem.replace("helm_", "")
        if name:
            discovered.add(name)
            logger.info(f"Discovered existing benchmark: {name}")

    discovered.update(DEFAULT_BENCHMARKS)
    logger.info(f"Default benchmarks: {DEFAULT_BENCHMARKS}")
    
    result = sorted(discovered)
    logger.info(f"Total benchmarks to process: {result}")
    return result


def ensure_csv_for_benchmark(benchmark: str) -> Path:
    """Ensure a `helm_{benchmark}.csv` exists; create it by scraping if missing.
    
    Returns the path to the CSV file.
    """
    csv_path = BENCHMARK_CSVS_DIR / f"helm_{benchmark}.csv"
    
    if csv_path.exists():
        logger.info(f"CSV already exists for benchmark '{benchmark}': {csv_path}")
        return csv_path
    
    logger.info(f"CSV missing for benchmark '{benchmark}', starting web scraping...")
    
    # Import scraper here to avoid import-time side effects
    from .web_scraper import scrape_benchmark_to_csv
    
    logger.info(f"Scraping HELM website for benchmark: {benchmark}")
    try:
        scrape_benchmark_to_csv(benchmark, csv_path)
        logger.info(f"✓ Successfully scraped benchmark '{benchmark}' to {csv_path}")
        
        # Log some stats about the scraped data
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            logger.info(f"  Scraped {len(df)} tasks for benchmark '{benchmark}'")
        
    except Exception as e:
        logger.error(f"✗ Failed to scrape benchmark '{benchmark}': {e}")
        raise
    
    return csv_path


def process_benchmark_csv(csv_path: Path, benchmark: str, downloads_dir: Path | None = None,
                          keep_temp: bool = False, overwrite: bool = False, max_workers: int = 8):
    """Call the project's unified `main.py` to download & convert all tasks in a CSV.

    The main.py expects: --source, --benchmark, --downloads-dir, etc.
    We reuse it to avoid duplicating download/convert logic.
    """
    logger.info(f"Starting to process benchmark CSV: {csv_path}")
    
    output_dir = PROCESSED_DATA_DIR / benchmark
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Run the processor script as a subprocess to avoid import-time side-effects
    python_exec = sys.executable
    cmd = [
        python_exec,
        str(Path.cwd() / "main.py"),
        "--source",
        "helm",
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

    logger.info(f"Running processor command: {' '.join(cmd)}")
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        
        if res.returncode != 0:
            logger.error(f"✗ Processor failed for benchmark '{benchmark}' (exit code {res.returncode})")
            logger.error(f"STDOUT: {res.stdout}")
            logger.error(f"STDERR: {res.stderr}")
            raise RuntimeError(f"Processor failed for benchmark '{benchmark}'")
        
        logger.info(f"✓ Processor completed successfully for benchmark '{benchmark}'")
        
        # Log some stats about what was processed
        if output_dir.exists():
            csv_files = list(output_dir.glob("*.csv"))
            logger.info(f"  Generated {len(csv_files)} processed CSV files")
            
            total_rows = 0
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    total_rows += len(df)
                except:
                    pass
            logger.info(f"  Total processed rows: {total_rows}")
        
    except Exception as e:
        logger.error(f"✗ Error running processor for benchmark '{benchmark}': {e}")
        raise


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


def aggregate_benchmark_to_parquet(output_parquet: Path, benchmark: str, input_dir: Path = PROCESSED_DATA_DIR,
                                  schema: pa.Schema | None = None, batch_size: int = 20):
    """Aggregate CSV files for a specific benchmark into a single Parquet file.

    - Only processes CSVs under `input_dir/{benchmark}/*.csv`
    - Ensures the resulting parquet conforms to `schema` (adds missing columns as null)
    - Writes a single file at `output_parquet`
    """
    logger.info(f"Starting aggregation of benchmark '{benchmark}' to parquet: {output_parquet}")
    
    if schema is None:
        schema = _build_pyarrow_schema()
    
    logger.info(f"Using schema with {len(schema)} fields")

    # Only look for CSVs in the specific benchmark directory
    benchmark_dir = input_dir / benchmark
    logger.info(f"Looking for CSV files in: {benchmark_dir}")
    
    csv_paths = sorted([p for p in benchmark_dir.rglob("*.csv") if p.is_file()])
    logger.info(f"Found {len(csv_paths)} CSV files to aggregate")
    
    if not csv_paths:
        error_msg = f"No processed CSV files found for benchmark '{benchmark}' under {benchmark_dir}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    out_dir = output_parquet.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")
    
    if output_parquet.exists():
        logger.info(f"Removing existing parquet file: {output_parquet}")
        output_parquet.unlink()

    writer: pq.ParquetWriter | None = None
    total_rows = 0

    for i in range(0, len(csv_paths), batch_size):
        batch = csv_paths[i: i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(csv_paths) + batch_size - 1)//batch_size} ({len(batch)} files)")
        frames = []
        
        for csv_p in batch:
            try:
                logger.info(f"  Reading CSV: {csv_p.name}")
                df = pd.read_csv(csv_p)
                logger.info(f"    Loaded {len(df)} rows")
            except Exception as e:
                logger.warning(f"  [skip] Failed reading {csv_p}: {e}")
                continue

            # Record provenance
            df["source_csv"] = str(csv_p.name)
            # Use the benchmark parameter instead of inferring from parent folder
            df["source_benchmark"] = benchmark

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
            # Use HuggingFace-optimized parquet settings
            # - SNAPPY compression for good balance of speed/size
            # - Row group size of 100k rows (HF recommendation)
            # - Write statistics for better querying performance
            writer = pq.ParquetWriter(
                output_parquet, 
                table.schema,
                compression='snappy',
                use_dictionary=True,
                write_statistics=True,
                row_group_size=100000  # 100k rows per row group (HF recommendation)
            )
        writer.write_table(table)
        total_rows += table.num_rows
        logger.info(f"✓ Wrote batch of {table.num_rows:,} rows (total so far: {total_rows:,})")

    if writer is not None:
        writer.close()
        logger.info(f"✓ Parquet writer closed. Final output: {output_parquet}")
    else:
        error_msg = f"No data written to parquet for benchmark '{benchmark}'. Check processed CSV files."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info(f"🎉 Successfully aggregated {total_rows:,} rows from benchmark '{benchmark}' -> {output_parquet}")
    return output_parquet


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
            # Use HuggingFace-optimized parquet settings
            # - SNAPPY compression for good balance of speed/size
            # - Row group size of 100k rows (HF recommendation)
            # - Write statistics for better querying performance
            writer = pq.ParquetWriter(
                output_parquet, 
                table.schema,
                compression='snappy',
                use_dictionary=True,
                write_statistics=True,
                row_group_size=100000  # 100k rows per row group (HF recommendation)
            )
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
    4. Aggregate processed CSVs into a Parquet file with schema
    
    If multiple benchmarks are specified, they will be aggregated together.
    If a single benchmark is specified, only that benchmark's data will be aggregated.
    """
    output_path = Path(output)
    downloads_path = Path(downloads_dir) if downloads_dir else None

    benchmarks_to_run = benchmarks if benchmarks else discover_benchmarks()
    print(f"Benchmarks to process: {benchmarks_to_run}")

    logger.info(f"🚀 Starting processing for {len(benchmarks_to_run)} benchmarks: {benchmarks_to_run}")

    for i, b in enumerate(benchmarks_to_run, 1):
        try:
            logger.info(f"\n=== Processing benchmark {i}/{len(benchmarks_to_run)}: '{b}' ===")
            csv_path = ensure_csv_for_benchmark(b)
            logger.info(f"Using CSV: {csv_path}")
            
            process_benchmark_csv(csv_path, b, downloads_dir=downloads_path,
                                  keep_temp=keep_temp, overwrite=overwrite, max_workers=max_workers)
            logger.info(f"✓ Benchmark '{b}' processed successfully")
            
        except Exception as e:
            logger.error(f"✗ Benchmark '{b}' failed: {e}")
            raise

    # Aggregate based on the number of benchmarks
    logger.info(f"\n=== Starting aggregation to parquet ===")
    if len(benchmarks_to_run) == 1:
        # Single benchmark - use benchmark-specific aggregation
        logger.info(f"Single benchmark mode - aggregating '{benchmarks_to_run[0]}' only")
        aggregate_benchmark_to_parquet(output_path, benchmarks_to_run[0])
    else:
        # Multiple benchmarks - use the original aggregation that combines all
        logger.info(f"Multi-benchmark mode - aggregating all {len(benchmarks_to_run)} benchmarks")
        aggregate_all_processed_to_parquet(output_path)
    
    logger.info(f"🎉 All processing completed! Final parquet: {output_path}")


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
