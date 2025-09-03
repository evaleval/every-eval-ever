# Every Eval Ever

A comprehensive, extensible data pipeline for collecting, processing, and serving evaluation datasets from multiple sources. Currently supports **HELM** with a modular architecture designed for easy addition of new evaluation sources.

## 📑 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📋 How This Repository Works](#-how-this-repository-works)  
- [🧩 Supported Sources](#-supported-sources)
- [📊 Data Pipeline Overview](#-data-pipeline-overview)
- [⚙️ Configuration](#️-configuration)
- [🤖 Automation](#-automation)
- [🔧 Adding New Data Sources](#-adding-new-data-sources)
- [💻 Development](#-development)
- [📈 Usage Examples](#-usage-examples)
- [🛠️ Output Format](#️-output-format)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙋 Support](#-support)

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/evaleval/every-eval-ever.git
cd every-eval-ever

# Install dependencies
pip install -r requirements.txt

# Install Playwright for web scraping
playwright install
```

### Basic Usage
```bash
# Process HELM data for the lite benchmark (fastest for testing)
python main.py helm --benchmark lite

# Process other HELM benchmarks
python main.py helm --benchmark mmlu
python main.py helm --benchmark classic

# Process with more parallelism
python main.py helm --benchmark lite --max-workers 8

# Keep temporary files for debugging
python main.py helm --benchmark lite --keep-temp --verbose
```

### Accessing Processed Data
The processed data is available on HuggingFace Hub:

```python
from datasets import load_dataset

# Load the complete dataset
dataset = load_dataset("evaleval/every_eval_ever")

# Stream for large datasets (recommended)
dataset = load_dataset("evaleval/every_eval_ever", streaming=True)

# Filter for specific sources
helm_data = dataset.filter(lambda x: x['source'] == 'helm')
```

## 📋 How This Repository Works

### 1. **Multi-Source Architecture**
Each evaluation source (like HELM) has its own processing pipeline while sharing common utilities:

```
src/
├── sources/                    # Source-specific processors
│   └── helm/                  # HELM evaluation data
│       ├── processor.py       # Main processing entry point
│       ├── web_scraper.py     # Scrapes HELM website for tasks
│       ├── downloader.py      # Downloads evaluation files
│       └── converter.py       # Converts to standardized format
├── core/                      # Shared processing utilities
│   ├── aggregator.py         # Combines data into Parquet shards
│   └── converter.py          # Base conversion utilities
└── utils/                     # Common utility functions
```

### 2. **Unified Command Interface**
All sources are accessed through a single entry point:

```bash
python main.py {source_name} [options]
```

Examples:
- `python main.py helm --benchmark lite`
- `python main.py openai_evals --dataset hellaswag` (future)
- `python main.py bigbench --subset reasoning` (future)

### 3. **Standardized Output Schema**
All sources produce data with the same schema for consistency:

```
evaluation_id          # Unique identifier
dataset_name           # Name of the dataset/task
model_name            # Model being evaluated
raw_input             # Input text/prompt
ground_truth          # Expected answer
output                # Model's actual output
evaluation_score      # Numerical score
source                # Source name (e.g., 'helm')
ingestion_timestamp   # When processed
# ... additional fields
```

### 4. **Automated Processing**
GitHub Actions automatically:
- Runs weekly data collection
- Processes new evaluations
- Creates optimized Parquet shards
- Uploads to HuggingFace Hub with incremental updates

## 🧩 Supported Sources

### HELM (Holistic Evaluation of Language Models)

Stanford's comprehensive language model evaluation framework.

- **Documentation**: [docs/HELM.md](docs/HELM.md)
- **Benchmarks**: `lite`, `classic`, `mmlu`, and many more
- **Status**: ✅ **Production Ready**

**Example Usage:**
```bash
# Quick test with lite benchmark
python main.py helm --benchmark lite

# Full MMLU evaluation
python main.py helm --benchmark mmlu --max-workers 4
```

## 📊 Data Pipeline Overview

1. **Discovery**: Scrape source websites to find available evaluations
2. **Download**: Fetch raw evaluation data and model outputs  
3. **Convert**: Transform into standardized schema
4. **Aggregate**: Combine into HuggingFace-optimized Parquet files
5. **Upload**: Deploy to HuggingFace Hub with incremental updates

Each step is modular and can be run independently for debugging.

## ⚙️ Configuration

The repository uses several configuration files:

- **`config/settings.py`**: Global settings and default values
- **`config/dataset_mappings.json`**: Maps source-specific fields to standard schema
- **`config/model_metadata.csv`**: Database of model information and families
- **`requirements.txt`**: Python dependencies

## 🤖 Automation

### GitHub Actions Workflow
The repository includes automated weekly processing:

```yaml
# .github/workflows/scrape_and_upload.yml
- Name: "Weekly Evaluation Data Processing"
- Schedule: Mondays at 03:00 UTC
- Current: HELM (lite, mmlu, classic benchmarks)
- Future: Multi-source parallel processing
- Output: Incremental Parquet shards
- Upload: Direct to HuggingFace Hub
```

### Manual Automation
For custom processing or testing:

```bash
# Run the automation script locally
python scripts/incremental_upload.py \
  --repo-id your-org/your-dataset \
  --benchmarks lite mmlu classic \
  --source-name helm \
  --max-workers 2
```

## 🔧 Adding New Data Sources

This repository is designed to make adding new evaluation sources straightforward. Follow this step-by-step guide:

### Step 1: Create Source Directory Structure
```bash
# Create the new source directory
mkdir src/sources/{your_source_name}
cd src/sources/{your_source_name}

# Create required files
touch __init__.py
touch processor.py      # Main entry point (required)
touch web_scraper.py   # Website scraping (if needed)
touch downloader.py    # Data downloading (if needed)  
touch converter.py     # Data conversion (if needed)
```

### Step 2: Implement Core Processor
Create `src/sources/{your_source_name}/processor.py`:

```python
#!/usr/bin/env python3
"""
{Your Source Name} evaluation data processor.
"""

import argparse
import logging
from pathlib import Path

# Import shared utilities
from src.utils.data_loading import save_to_parquet
from src.utils.evaluation_utils import standardize_evaluation_data

logger = logging.getLogger(__name__)

def main():
    """Main entry point for {your_source_name} processing."""
    parser = argparse.ArgumentParser(description="Process {Your Source Name} evaluation data")
    
    # Add your source-specific arguments
    parser.add_argument("--benchmark", help="Benchmark name to process")
    parser.add_argument("--output", help="Output parquet file path")
    parser.add_argument("--max-workers", type=int, default=4, help="Max parallel workers")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    try:
        # Your processing logic here
        logger.info(f"Processing {args.benchmark} benchmark...")
        
        # 1. Scrape/discover available evaluations
        # evaluations = discover_evaluations(args.benchmark)
        
        # 2. Download raw data
        # raw_data = download_data(evaluations, args.max_workers)
        
        # 3. Convert to standard schema
        # standardized_data = convert_to_schema(raw_data)
        
        # 4. Save as Parquet
        # save_to_parquet(standardized_data, args.output)
        
        logger.info("✅ Processing completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
```

### Step 3: Add Source to Main Router
Edit `main.py` to include your new source:

```python
# Add to SUPPORTED_SOURCES at the top
SUPPORTED_SOURCES = {
    'helm': 'src.sources.helm.processor',
    'your_source_name': 'src.sources.your_source_name.processor',  # Add this line
}
```

### Step 4: Create Documentation
Create `docs/{YOUR_SOURCE_NAME}.md` with:

```markdown
# {Your Source Name} Data Source

Description of your evaluation source...

## Quick Start
```bash
python main.py your_source_name --benchmark example
```

## Available Benchmarks
- `benchmark1`: Description
- `benchmark2`: Description

## Data Pipeline Details
Explain your specific processing steps...

## Common Issues and Solutions
Document troubleshooting steps...
```

### Step 5: Update Configuration
Add your source to `config/dataset_mappings.json`:

```json
{
  "your_source_name": {
    "field_mappings": {
      "source_field_name": "standard_field_name"
    },
    "default_values": {
      "source": "your_source_name",
      "license": "your_license"
    }
  }
}
```

### Step 6: Add to Automation
Update `scripts/incremental_upload.py` to support your source:

```python
# In main() function, modify the supported sources check
SUPPORTED_SOURCES = ['helm', 'your_source_name']
```

Update GitHub Actions workflow to include your source:

```yaml
# In .github/workflows/scrape_and_upload.yml
- name: Process evaluation data and upload to HuggingFace
  run: |
    python scripts/incremental_upload.py \
      --repo-id evaleval/every_eval_ever \
      --benchmarks your_benchmark1 your_benchmark2 \
      --source-name your_source_name
```

### Step 7: Test Your Integration
```bash
# Test basic functionality
python main.py your_source_name --help

# Test with small dataset
python main.py your_source_name --benchmark test --verbose

# Test automation script
python scripts/incremental_upload.py \
  --repo-id test/dataset \
  --benchmarks test \
  --source-name your_source_name \
  --max-workers 1
```

### Step 8: Schema Compliance
Ensure your data matches the standardized schema:

**Required Fields:**
- `evaluation_id` (string): Unique identifier
- `dataset_name` (string): Name of the task/dataset
- `model_name` (string): Model being evaluated
- `raw_input` (string): Input text/prompt
- `ground_truth` (string): Expected answer
- `output` (string): Model's actual output
- `evaluation_score` (float): Numerical score
- `source` (string): Your source name
- `ingestion_timestamp` (datetime): When processed

**Optional Fields:**
- `model_family`, `hf_split`, `hf_index`, `evaluation_method_name`, etc.

### Example: OpenAI Evals Integration
Here's a complete example for adding OpenAI Evals:

```bash
# 1. Create structure
mkdir src/sources/openai_evals
touch src/sources/openai_evals/{__init__.py,processor.py,converter.py}

# 2. Implement processor.py
# (See template above)

# 3. Add to main.py
# 'openai_evals': 'src.sources.openai_evals.processor'

# 4. Test
python main.py openai_evals --help
```

### Best Practices for New Sources

1. **Follow the Pattern**: Use existing HELM implementation as reference
2. **Error Handling**: Include comprehensive error handling and logging
3. **Rate Limiting**: Respect source websites' rate limits
4. **Resumability**: Support resuming interrupted downloads
5. **Documentation**: Provide clear usage examples and troubleshooting
6. **Testing**: Include unit tests for critical functions
7. **Schema Compliance**: Ensure data matches the standard schema exactly

### Getting Help

- **Reference Implementation**: Check `src/sources/helm/` for a complete example
- **Schema Details**: See `src/utils/evaluation_utils.py` for schema validation
- **Shared Utilities**: Use functions in `src/core/` and `src/utils/` when possible
- **Issues**: Open a GitHub issue for guidance on specific sources

## 💻 Development

### Local Development
```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .  # Install in development mode

# Run tests
python -m pytest tests/ -v

# Check code style
black . && isort .

# Type checking
mypy src/

# Process specific benchmark for testing
python main.py helm --benchmark lite --max-workers 4 --verbose

# Local incremental processing (without upload)
python scripts/incremental_upload.py \
  --benchmarks lite \
  --source-name helm \
  --max-workers 2 \
  --repo-id test/dataset
```

### Testing Individual Components
For debugging specific parts of the pipeline:

```bash
# Test HELM web scraper
python -m src.sources.helm.web_scraper --benchmark test

# Test HELM downloader with specific task
python -m src.sources.helm.downloader --task sample_task

# Test converter with sample data
python -m src.sources.helm.converter \
  --input data/downloads/sample \
  --output test.csv
```

### Debugging Tips
1. **Use `--verbose`** for detailed logging
2. **Use `--keep-temp`** to inspect intermediate files
3. **Use `--max-workers 1`** to avoid parallel processing issues
4. **Check logs** in `{source}_processor.log` files

## 📈 Usage Examples

### Research Analysis
```python
from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset("evaleval/every_eval_ever", streaming=True)

# Convert to pandas for analysis
df = dataset.to_pandas()

# Compare model performance across datasets
performance = df.groupby(['model_name', 'dataset_name'])['evaluation_score'].mean()

# Find best performing models
top_models = df.groupby('model_name')['evaluation_score'].mean().sort_values(ascending=False)

# Analysis by evaluation method
method_analysis = df.groupby(['evaluation_method_name', 'model_name'])['evaluation_score'].mean()
```

### Model Evaluation Comparison
```python
# Filter for specific models
gpt_models = df[df['model_name'].str.contains('gpt', case=False)]

# Compare across datasets
comparison = gpt_models.pivot_table(
    values='evaluation_score',
    index='dataset_name', 
    columns='model_name',
    aggfunc='mean'
)

# Statistical analysis
from scipy import stats
model_a_scores = df[df['model_name'] == 'gpt-4']['evaluation_score']
model_b_scores = df[df['model_name'] == 'claude-2']['evaluation_score']
statistic, p_value = stats.ttest_ind(model_a_scores, model_b_scores)
```

### Time Series Analysis
```python
# Track evaluation trends over time
df['ingestion_date'] = pd.to_datetime(df['ingestion_timestamp']).dt.date

timeline = df.groupby(['ingestion_date', 'model_name'])['evaluation_score'].mean().reset_index()

# Plot trends
import matplotlib.pyplot as plt
for model in timeline['model_name'].unique():
    model_data = timeline[timeline['model_name'] == model]
    plt.plot(model_data['ingestion_date'], model_data['evaluation_score'], label=model)

plt.legend()
plt.title('Model Performance Over Time')
plt.xlabel('Date')
plt.ylabel('Average Evaluation Score')
plt.show()
```

## 🛠️ Output Format

The processed data follows a consistent structure optimized for HuggingFace Datasets:

### File Naming Convention
```
data-00001-helm.parquet      # First HELM shard  
data-00002-helm.parquet      # Second HELM shard
data-00003-openai.parquet    # First OpenAI Evals shard (future)
data-00004-bigbench.parquet  # First BigBench shard (future)
```

### Parquet Optimization
- **Compression**: SNAPPY for fast read/write
- **Row Groups**: 100,000 rows per group for optimal streaming
- **Schema**: Consistent across all sources
- **Size**: Approximately 50-100MB per shard

### Data Schema
Each record contains standardized evaluation information:

```python
{
  'evaluation_id': 'helm_mmlu_gpt4_001',
  'dataset_name': 'mmlu_anatomy', 
  'model_name': 'gpt-4',
  'model_family': 'gpt',
  'raw_input': 'What is the largest organ in the human body?',
  'ground_truth': 'skin',
  'output': 'The skin is the largest organ...',
  'evaluation_score': 0.95,
  'evaluation_method_name': 'exact_match',
  'source': 'helm',
  'source_version': '1.0',
  'ingestion_timestamp': '2024-01-15T10:30:00Z',
  'license': 'apache-2.0',
  'category': 'knowledge'
}
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Types of Contributions
1. **New Data Sources**: Add support for additional evaluation frameworks
2. **Bug Fixes**: Fix issues in existing processing pipelines  
3. **Performance Improvements**: Optimize data processing and storage
4. **Documentation**: Improve guides and API documentation
5. **Testing**: Add comprehensive test coverage

### Contribution Process
1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/new-source`
3. **Implement** your changes with tests and documentation
4. **Test** thoroughly with sample data
5. **Submit** a pull request with detailed description

### Code Standards
- **Style**: Use `black` and `isort` for formatting
- **Types**: Add type hints for public functions
- **Tests**: Include unit tests for new functionality
- **Docs**: Update relevant documentation
- **Logging**: Use structured logging with appropriate levels

### Testing New Sources
Before submitting a new source integration:

```bash
# Test the integration
python main.py your_source --benchmark test --verbose

# Validate schema compliance
python -c "
from datasets import Dataset
import pandas as pd
df = pd.read_parquet('your_output.parquet')
dataset = Dataset.from_pandas(df)
print('✅ Schema validation passed')
"

# Test automation integration
python scripts/incremental_upload.py \
  --benchmarks test \
  --source-name your_source \
  --repo-id test/dataset
```

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙋 Support

- **Issues**: [GitHub Issues](https://github.com/evaleval/every-eval-ever/issues)
- **Discussions**: [GitHub Discussions](https://github.com/evaleval/every-eval-ever/discussions)  
- **Dataset**: [HuggingFace Dataset Page](https://huggingface.co/datasets/evaleval/every_eval_ever)
- **Documentation**: Source-specific docs in `docs/` directory

### FAQ

**Q: How do I access just HELM data?**
```python
dataset = load_dataset("evaleval/every_eval_ever")
helm_data = dataset.filter(lambda x: x['source'] == 'helm')
```

**Q: Can I run processing without uploading to HuggingFace?**
```bash
python main.py helm --benchmark lite --output local_output.parquet
```

**Q: How do I add a custom evaluation source?**
See the [Adding New Data Sources](#-adding-new-data-sources) section above.

**Q: What's the difference between benchmarks and datasets?**
- **Benchmark**: A collection of tasks (e.g., "lite", "mmlu")
- **Dataset**: Individual tasks within benchmarks (e.g., "hellaswag", "winogrande")

---

**🎯 Goal**: Create the most comprehensive, accessible, and standardized evaluation dataset for language model research.

**💡 Vision**: Enable researchers to easily compare models across all major evaluation frameworks through a single, unified interface.
