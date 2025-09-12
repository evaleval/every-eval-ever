# Every Eval Ever

A comprehensive data pipeline for collecting, processing, and serving evaluation datasets from multiple sources using the standardized [EvalHub schema](https://github.com/evaleval/evalHub).

## 🚀 Quick Start

```bash
# Setup
git clone https://github.com/evaleval/every-eval-ever.git
cd every-eval-ever
pip install -r requirements.txt
./setup_schemas.sh

# Test processing
python scripts/simple_helm_processor_evalhub.py --test-run

# Full processing (requires HF_TOKEN)
python scripts/simple_helm_processor_evalhub.py --repo-id evaleval/every_eval_ever
```

## 📊 Datasets

| Dataset | HuggingFace Link | Description |
|---------|------------------|-------------|
| **Evaluations** | [evaleval/every_eval_ever](https://huggingface.co/datasets/evaleval/every_eval_ever) | Individual evaluation results in EvalHub format |
| **Statistics** | [evaleval/every_eval_score_ever](https://huggingface.co/datasets/evaleval/every_eval_score_ever) | Aggregated performance statistics |

## 🧩 Supported Sources

- **[HELM](docs/HELM.md)** - Holistic Evaluation of Language Models
- *More sources coming soon...*

## 📋 Schema

All data follows the [EvalHub schema](https://github.com/evaleval/evalHub) with nested evaluation objects:

```json
{
  "evaluation_id": "unique_eval_id",
  "benchmark": "mmlu",
  "model_info": {
    "model_name": "gpt-4",
    "model_family": "openai"
  },
  "instance_result": {
    "instance_id": 123,
    "input": "Question text",
    "output": "Answer text", 
    "score": 0.85,
    "is_correct": true
  },
  "source": "helm"
}
```

## 🔧 Usage

### Load Dataset
```python
from datasets import load_dataset

# Load evaluations
dataset = load_dataset("evaleval/every_eval_ever")
df = dataset['train'].to_pandas()

# Analyze performance
performance = df.groupby(['model_info.model_name', 'benchmark'])['instance_result.score'].mean()
```

### Add New Source
1. Create processor in `scripts/your_source_processor.py`
2. Follow EvalHub schema format
3. Add documentation in `docs/YOUR_SOURCE.md`
4. Update workflow in `.github/workflows/`

## 📁 Structure

```
every-eval-ever/
├── scripts/
│   └── simple_helm_processor_evalhub.py  # HELM processor
├── docs/
│   └── HELM.md                           # HELM-specific docs
├── tests/                                # Test suite
├── config/                               # Configuration files
└── data/                                 # Processed data (local)
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/new-source`
3. **Follow** EvalHub schema in your processor
4. **Add** tests and documentation
5. **Submit** pull request

## 📄 License

MIT License - see LICENSE file for details.

---

For source-specific documentation, see the `docs/` directory.
