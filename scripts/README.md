# Scripts Directory

This directory contains automation scripts for the Every Eval Ever project.

## incremental_upload.py

Main script for the weekly GitHub Actions automation. Handles:

- 🔍 **Discovery**: Finds the next part number from existing HuggingFace dataset
- 🔄 **Processing**: Runs HELM scraping and processing for each benchmark incrementally
- ☁️ **Upload**: Immediately uploads each parquet shard to HuggingFace
- 🧹 **Cleanup**: Removes local files to save disk space
- 📝 **Logging**: Comprehensive logging with timestamps and progress indicators

### Usage

```bash
# Basic usage (used by GitHub Actions)
python scripts/incremental_upload.py \
  --repo-id evaleval/every_eval_ever \
  --benchmarks lite mmlu classic

# Custom configuration
python scripts/incremental_upload.py \
  --repo-id your-org/your-dataset \
  --benchmarks lite mmlu classic custom \
  --source-name helm \
  --max-workers 4 \
  --timeout 7200
```

### Arguments

- `--repo-id`: HuggingFace dataset repository ID (required)
- `--benchmarks`: List of HELM benchmarks to process (default: lite mmlu classic)
- `--source-name`: Source identifier for file naming (default: helm)
- `--max-workers`: Maximum parallel workers for processing (default: 2)
- `--timeout`: Timeout per benchmark in seconds (default: 3600)

### Environment Variables

- `HF_TOKEN`: HuggingFace API token (required)

### Logs

The script generates logs in:
- **Console**: Real-time progress with colored indicators
- **File**: `incremental_upload.log` for debugging

### Features

- 🔄 **Incremental Processing**: Each benchmark becomes a separate parquet shard
- 📊 **Progress Tracking**: Shows X/Y benchmark progress and file statistics  
- ⚡ **Real-time Output**: Streams subprocess output as it happens
- 🛡️ **Error Handling**: Continues processing other benchmarks if one fails
- 📈 **Smart Numbering**: Automatically continues from the last part number
- 🎯 **Memory Efficient**: Processes and uploads one benchmark at a time
