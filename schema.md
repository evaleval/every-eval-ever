# Final Parquet Schema for HELM Aggregation

This file documents the Parquet schema used by `src/core/scrape_all_to_parquet.py` when it
aggregates all processed HELM CSVs into a single Parquet file.

All fields are nullable unless explicitly noted.

Fields and types:

- evaluation_id (string)
  - Unique identifier for an individual evaluation example. Produced by `create_evaluation_id` in `src/utils/model_utils.py`.

- dataset_name (string)
  - Normalized dataset identifier (e.g., `mmlu.anatomy`). Extracted from HELM `run_spec` and `scenario`.

- hf_split (string)
  - Dataset split label (commonly `train`, `test`, `validation`). Normalized so `valid` -> `validation`.

- hf_index (int64)
  - Integer index of the example in the original dataset/split. Use - or null for missing indices.

- raw_input (string)
  - The original prompt or question presented to the model.

- ground_truth (string)
  - The expected answer or label for the example (for multiple choice this may be `A`, `B`, etc.).

- model_name (string)
  - Canonical model identifier (e.g., `gpt-3.5-turbo`), mapped from HELM run metadata.

- model_family (string)
  - Higher-level model family or architecture (e.g., `gpt-3.5`, `bert-family`).

- output (string)
  - Model-generated output/prediction for the example.

- evaluation_method_name (string)
  - The name of the evaluation metric used (e.g., `exact_match`, `f1_score`).

- evaluation_score (float64)
  - Numerical score (0-1) produced by the evaluation metric.

Extra provenance / metadata fields included in the Parquet file:

- run (string)
  - HELM "Run" label from the benchmark CSV table (sometimes contains model/run identifier).

- task (string)
  - Human-friendly task name or subject from the HELM benchmark lines.

- adapter_method (string)
  - Adapter or prompting method used (if present in the benchmark CSV row).

# Final Parquet Schema for Aggregated Evaluations

This file documents the Parquet schema used by `src/core/scrape_all_to_parquet.py` when it
aggregates processed evaluation CSVs (initially from HELM) into a single Parquet file.

All fields are nullable unless explicitly noted.

Primary evaluation fields and types:

- evaluation_id (string)
  - Unique identifier for an individual evaluation example. Produced by repository tooling (e.g., `create_evaluation_id`).

- dataset_name (string)
  - Normalized dataset identifier (e.g., `mmlu.anatomy`).

- hf_split (string)
  - Dataset split label (commonly `train`, `test`, `validation`).

- hf_index (int64)
  - Integer index of the example in the original dataset/split.

- raw_input (string)
  - The original prompt or question presented to the model.

- ground_truth (string)
  - The expected answer or label for the example (for multiple choice this may be `A`, `B`, etc.).

- model_name (string)
  - Canonical model identifier (e.g., `gpt-3.5-turbo`).

- model_family (string)
  - Higher-level model family or architecture.

- output (string)
  - Model-generated output/prediction for the example.

- evaluation_method_name (string)
  - The name of the evaluation metric used (e.g., `exact_match`, `f1_score`).

- evaluation_score (float64)
  - Numerical score (0-1) produced by the evaluation metric.

Provenance / metadata fields included in the Parquet file:

- run (string)
  - HELM "Run" label or equivalent run identifier for other sources.

- task (string)
  - Human-friendly task name or subject.

- adapter_method (string)
  - Adapter or prompting method used (if present in the source CSV row).

- source (string)
  - The origin of the evaluation data (e.g., `helm`, `other_source`). Required to support multiple sources.

- source_version (string)
  - Version identifier (if available) for the upstream source (e.g., a HELM release tag).

- source_url (string)
  - A URL or identifier pointing to the upstream data location (if available).

- ingestion_timestamp (timestamp)
  - UTC timestamp (ns resolution) when the row was written into the aggregated Parquet file.

- license (string)
  - Upstream license or usage terms for this evaluation data if known.

- category (string)
  - Reserved field for tagging the evaluation with one of 20 risk/capability categories. Leave blank for now.

- source_csv (string)
  - The original processed CSV filename within `data/processed/{benchmark}/` from which this row was read.

- source_benchmark (string)
  - The benchmark folder (e.g., `lite`, `mmlu`, `classic`) under `data/processed/` that contained the CSV.

Notes / Implementation details:

- The Parquet schema is defined in code using `pyarrow.Schema` in `src/core/scrape_all_to_parquet.py::_build_pyarrow_schema()`.
- During aggregation, any missing columns will be added with null values to match the schema.
- The aggregator sets `source` to `helm` for existing processed CSVs and populates `ingestion_timestamp`.

If you update field names or types in the converter pipeline (`src/core/converter.py`), update the
schema in `_build_pyarrow_schema()` and this file.
