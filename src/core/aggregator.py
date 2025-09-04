from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import PROCESSED_DATA_DIR, AGGREGATED_DATA_DIR

REQUIRED_COLUMNS: List[str] = [
    "dataset_name",
    "hf_split", 
    "hf_index",
    "model_name",
    "model_family",
    "evaluation_method_name",
    "evaluation_score",
]

# Additional metadata columns for multi-source support
METADATA_COLUMNS: List[str] = [
    "evaluation_id",
    "raw_input",
    "ground_truth", 
    "output",
]

# All columns to keep in the final dataset
ALL_COLUMNS = REQUIRED_COLUMNS + METADATA_COLUMNS


def _batch_iter(items: List[Path], batch_size: int) -> Iterable[List[Path]]:
    for i in range(0, len(items), batch_size):
        yield items[i: i + batch_size]


def aggregate_to_parquet(
        benchmark: str,
        input_dir: Path = PROCESSED_DATA_DIR,
        output_dir: Path = AGGREGATED_DATA_DIR,
        batch_size: int = 10,
        source_name: str = "helm",
) -> Path:
    """Aggregate many HELM CSVs into a single Parquet file.

    - Reads files in batches of `batch_size` to keep memory bounded
    - Keeps required columns plus additional metadata
    - Normalizes hf_split: 'valid' -> 'validation'
    - Adds source metadata for multi-source support
    
    Args:
        benchmark: Benchmark type (e.g., 'classic', 'lite', 'mmlu')
        input_dir: Directory containing CSV files (default: data/processed/)
        output_dir: Directory to save parquet file (default: data/aggregated/)
        batch_size: Number of CSV files to process in each batch
        source_name: Name of the evaluation source (default: 'helm')
    """

    # Use benchmark-specific input directory
    in_dir = input_dir / benchmark

    # Create output filename and path
    filename = f"{source_name}_{benchmark}_aggregated.parquet"
    out_path = output_dir / filename

    in_dir = in_dir.resolve()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    csv_files = sorted([p for p in in_dir.glob("*.csv") if p.is_file()])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {in_dir}")

    writer: pq.ParquetWriter | None = None
    total_rows = 0

    for batch in _batch_iter(csv_files, batch_size):
        frames: List[pd.DataFrame] = []
        for csv_path in batch:
            try:
                # Read all available columns, then filter to what we want
                df = pd.read_csv(csv_path)
                
                # Keep only columns that exist in both ALL_COLUMNS and the CSV
                available_columns = [col for col in ALL_COLUMNS if col in df.columns]
                df = df[available_columns]
                
            except Exception as e:
                print(f"[skip] Failed reading {csv_path.name}: {e}")
                continue

            # Normalize split naming
            if "hf_split" in df.columns:
                df["hf_split"] = df["hf_split"].replace({"valid": "validation"})

            # Add source metadata
            df["source"] = source_name
            
            # Add comprehensive timestamp metadata
            from datetime import datetime, timezone
            current_time = datetime.now(timezone.utc)
            df["processed_at"] = current_time.isoformat()
            df["aggregation_timestamp"] = current_time.isoformat()
            df["pipeline_stage"] = "aggregation"

            frames.append(df)

        if not frames:
            continue

        batch_df = pd.concat(frames, ignore_index=True)

        # Define final column order with new metadata columns
        final_columns = ["source", "processed_at", "aggregation_timestamp", "pipeline_stage"] + [col for col in ALL_COLUMNS if col in batch_df.columns]
        batch_df = batch_df.reindex(columns=final_columns)

        table = pa.Table.from_pandas(batch_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        total_rows += len(batch_df)
        print(f"Wrote batch of {len(batch_df):,} rows (total {total_rows:,})")

    if writer is not None:
        writer.close()
    else:
        raise RuntimeError("No data written. Check input CSV files.")

    print(f"Done. Aggregated {total_rows:,} rows -> {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate CSV files into Parquet format for multi-source evaluation data")
    parser.add_argument("--benchmark", required=True, help="Benchmark type (e.g., 'classic', 'lite', 'mmlu')")
    parser.add_argument("--source-name", default="helm", help="Name of the evaluation source (default: 'helm')")
    parser.add_argument("--input-dir",
                        default=str(PROCESSED_DATA_DIR),
                        help=f"Input directory containing CSV files (default: {PROCESSED_DATA_DIR})")
    parser.add_argument("--output-dir",
                        default=str(AGGREGATED_DATA_DIR),
                        help=f"Output directory for parquet file (default: {AGGREGATED_DATA_DIR})")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing (default: 10)")

    args = parser.parse_args()

    # Convert string paths to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    aggregate_to_parquet(
        benchmark=args.benchmark,
        input_dir=input_dir,
        output_dir=output_dir,
        batch_size=args.batch_size,
        source_name=args.source_name
    )
