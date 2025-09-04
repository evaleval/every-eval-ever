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
The processed data is available on HuggingFace Hub in two datasets:

#### Detailed Evaluation Data
```python
from datasets import load_dataset

# Load the complete detailed dataset (individual evaluation records)
dataset = load_dataset("evaleval/every_eval_ever")

# Stream for large datasets (recommended)
dataset = load_dataset("evaleval/every_eval_ever", streaming=True)

# Filter for specific sources
helm_data = dataset.filter(lambda x: x['source'] == 'helm')
```

#### Comprehensive Statistics (Leaderboard Data)
```python
# Load benchmark-model performance statistics
stats_dataset = load_dataset("evaleval/every_eval_score_ever")

# Get top performers across all benchmarks
top_performers = stats_dataset['train'].sort('accuracy', reverse=True)

# Filter for specific benchmarks
gsm8k_stats = stats_dataset['train'].filter(lambda x: x['dataset_name'] == 'gsm8kscenario')
```

#### Data Overview
- **`evaleval/every_eval_ever`**: Detailed individual evaluation records (~50MB per benchmark)
- **`evaleval/every_eval_score_ever`**: Comprehensive benchmark-model statistics (~16KB total)
- **Update Frequency**: Data weekly, statistics daily
- **Schema**: Standardized across all evaluation sources

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

### 4. **Automated Dual-Dataset Processing**
GitHub Actions automatically:
- Runs weekly data collection and processing
- Creates detailed evaluation records in Parquet shards
- Uploads to `evaleval/every_eval_ever` with incremental updates
- Generates comprehensive statistics daily via incremental processing
- Uploads leaderboard data to `evaleval/every_eval_score_ever`
- Provides both granular data and aggregated insights

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

### Primary Pipeline (Weekly)
1. **Discovery**: Scrape source websites to find available evaluations
2. **Download**: Fetch raw evaluation data and model outputs  
3. **Convert**: Transform into standardized schema
4. **Aggregate**: Combine into HuggingFace-optimized Parquet files
5. **Upload**: Deploy detailed data to `evaleval/every_eval_ever`

### Statistics Pipeline (Daily)
1. **Download**: Fetch all data shards incrementally (memory-efficient)
2. **Aggregate**: Calculate comprehensive benchmark-model statistics
3. **Upload**: Deploy leaderboard data to `evaleval/every_eval_score_ever`

Each step is modular and can be run independently for debugging. The dual pipeline approach ensures both detailed research data and quick leaderboard access.

## ⚙️ Configuration

The repository uses several configuration files:

- **`config/settings.py`**: Global settings and default values
- **`config/dataset_mappings.json`**: Maps source-specific fields to standard schema
- **`config/model_metadata.csv`**: Database of model information and families
- **`requirements.txt`**: Python dependencies

## 🤖 Automation

The repository uses **dual workflow architecture** for optimal modularity and resource management:

### 📋 Workflow 1: Data Processing & Upload
```yaml
# .github/workflows/scrape_and_upload.yml
- Name: "Weekly Evaluation Data Processing"
- Schedule: Mondays at 03:00 UTC
- Purpose: Process and upload detailed evaluation data
- Current: HELM (lite, mmlu, classic benchmarks)
- Future: Multi-source parallel processing
- Output: Detailed evaluation records + per-shard statistics
- Upload: evaleval/every_eval_ever
- Timeout: 6 hours (for heavy data processing)
```

### 📊 Workflow 2: Comprehensive Statistics Generation
```yaml
# .github/workflows/generate_comprehensive_stats.yml
- Name: "Generate Comprehensive Statistics"
- Schedule: Daily at 06:00 UTC (3 hours after data processing)
- Purpose: Create unified leaderboard across all benchmarks
- Method: Incremental processing (memory-efficient)
- Output: Comprehensive benchmark-model statistics
- Upload: evaleval/every_eval_score_ever
- Timeout: 2 hours (for lighter aggregation)
- Manual Trigger: With source selection and force regenerate options
```

### 🎯 Benefits of Dual Architecture
- **Failure Isolation**: Data and stats workflows are independent
- **Resource Optimization**: Each workflow uses appropriate timeouts
- **Flexible Scheduling**: Weekly data collection, daily stats updates
- **Manual Control**: Trigger stats generation anytime with custom parameters

### Manual Data Processing
For custom processing or testing:

```bash
# Run the data processing pipeline locally
python scripts/incremental_upload.py \
  --repo-id your-org/your-dataset \
  --stats-repo-id your-org/your-stats-dataset \
  --benchmarks lite mmlu classic \
  --source-name helm \
  --max-workers 2
```

### Manual Statistics Generation
For generating comprehensive statistics:

```bash
# Generate comprehensive statistics from all uploaded data
python scripts/generate_comprehensive_stats.py \
  --main-repo-id evaleval/every_eval_ever \
  --stats-repo-id evaleval/every_eval_score_ever \
  --source-name helm
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

Update both GitHub Actions workflows to include your source:

```yaml
# In .github/workflows/scrape_and_upload.yml (Data Processing)
- name: Process evaluation data and upload to HuggingFace
  run: |
    python scripts/incremental_upload.py \
      --repo-id evaleval/every_eval_ever \
      --stats-repo-id evaleval/every_eval_score_ever \
      --benchmarks your_benchmark1 your_benchmark2 \
      --source-name your_source_name

# In .github/workflows/generate_comprehensive_stats.yml (Statistics)
- name: Generate comprehensive statistics
  run: |
    python scripts/generate_comprehensive_stats.py \
      --main-repo-id evaleval/every_eval_ever \
      --stats-repo-id evaleval/every_eval_score_ever \
      --source-name your_source_name
```

**Note**: The dual workflow architecture automatically handles your new source:
- **Data workflow**: Processes and uploads your detailed evaluation data
- **Stats workflow**: Includes your data in comprehensive statistics generation

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

The processed data is delivered through **two complementary datasets** optimized for different use cases:

### Dataset 1: Detailed Evaluation Records (`evaleval/every_eval_ever`)

**Purpose**: Individual evaluation instances for deep analysis and research

#### File Naming Convention
```
helm_lite_part_001.parquet       # HELM lite benchmark, part 1
helm_mmlu_part_002.parquet       # HELM mmlu benchmark, part 2  
helm_classic_part_003.parquet    # HELM classic benchmark, part 3
stats-helm_lite_part_001.parquet # Per-shard statistics
```

#### Parquet Optimization
- **Compression**: SNAPPY for fast read/write
- **Row Groups**: 100,000 rows per group for optimal streaming
- **Schema**: Consistent across all sources
- **Size**: Approximately 50MB per shard

#### Data Schema
Each record contains standardized evaluation information with rich metadata and comprehensive timestamps:

```python
{
  # Core evaluation data
  'dataset_name': 'openbookqa',
  'hf_split': 'test', 
  'hf_index': 4957,
  'model_name': '01-ai/yi-34b',
  'model_family': 'yi-34b',
  'evaluation_method_name': 'label_only_match',
  'evaluation_score': 1.0,
  
  # Rich metadata for analysis
  'evaluation_id': 'bb232a7d24e9c049bf1a8bfbfb92192010e0e3b8585aab6c5f147507837dd2af',
  'raw_input': 'A person wants to start saving money so that they can afford a nice vacation...',
  'ground_truth': 'B',
  'output': 'B',
  
  # Multi-source provenance and timestamps
  'source': 'helm',
  'processed_at': '2025-09-03T21:20:51.606743Z',
  'aggregation_timestamp': '2025-09-03T21:20:51.606743Z',
  'pipeline_stage': 'aggregation'
}
```

### Dataset 2: Comprehensive Statistics (`evaleval/every_eval_score_ever`)

**Purpose**: Benchmark-model performance summaries for leaderboards and quick analysis

#### File Naming Convention
```
comprehensive_stats_helm_20250903_120000.parquet  # Comprehensive stats with timestamp
stats-helm_lite_part_001.parquet                 # Individual shard stats (also here)
```

#### Statistics Schema
Each record represents a benchmark-model combination with aggregated metrics and comprehensive timestamps:

```python
{
  # Identification
  'source': 'helm',
  'dataset_name': 'gsm8kscenario',
  'model_name': 'openai/gpt-4-0613',
  'model_family': 'gpt-4',
  'evaluation_method_name': 'label_only_match',
  
  # Performance metrics
  'total_samples': 1000,
  'accuracy': 0.932,           # 93.2%
  'mean_score': 0.932,
  'std_score': 0.251746,
  'min_score': 0.0,
  'max_score': 1.0,
  
  # Metadata and timestamps
  'processed_at': '2025-09-03T22:15:30.123456Z',
  'statistics_generated_at': '2025-09-03T22:15:30.123456Z',
  'comprehensive_stats_generated_at': '2025-09-03T22:15:30.123456Z',
  'pipeline_stage': 'comprehensive_statistics',
  'data_freshness_hours': 0
}
```

### Dataset Comparison

| Aspect | Detailed Records | Comprehensive Statistics |
|--------|------------------|-------------------------|
| **Use Case** | Research, debugging, deep analysis | Leaderboards, quick comparisons |
| **Record Type** | Individual evaluations | Benchmark-model summaries |
| **File Size** | ~50MB per benchmark | ~16KB total |
| **Update Freq** | Weekly | Daily |
| **Memory Usage** | High (streaming recommended) | Low (fits in memory) |
| **Query Speed** | Slower (large data) | Faster (small data) |

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
