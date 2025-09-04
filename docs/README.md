# Documentation Index

## 📚 Architecture Documentation

### Core Architecture
- **[Dual Workflow Architecture](architecture/DUAL_WORKFLOW_ARCHITECTURE.md)** - Overview of the two-workflow system (data processing + statistics generation)
- **[Statistics Summary](architecture/STATISTICS_SUMMARY.md)** - Complete statistics pipeline implementation details
- **[Incremental Stats Solution](architecture/INCREMENTAL_STATS_SOLUTION.md)** - Memory-efficient statistics calculation approach
- **[Timestamp Enhancement](architecture/TIMESTAMP_ENHANCEMENT.md)** - Comprehensive timestamp tracking implementation

### Performance Optimization
- **[Optimization Guide](../OPTIMIZATION_GUIDE.md)** - Performance improvements and chunked processing (3-4x speedup)

### Data Sources
- **[HELM Integration](HELM.md)** - HELM data source documentation

## 🏗️ System Overview

The Every Eval Ever pipeline consists of:

1. **Data Processing Workflow** (Weekly)
   - Scrapes evaluation data from sources (HELM, etc.)
   - Processes and uploads detailed evaluation records
   - Optimized chunked processing for 3-4x performance improvement

2. **Statistics Generation Workflow** (Daily)
   - Downloads processed data incrementally
   - Generates comprehensive leaderboard statistics
   - Memory-efficient shard-based processing

3. **Dual Dataset Architecture**
   - `evaleval/every_eval_ever` - Detailed evaluation instances
   - `evaleval/every_eval_score_ever` - Benchmark-model summary statistics

## 🚀 Key Features

- **Performance**: 3-4x faster processing with chunked parallel uploads
- **Memory Efficiency**: Incremental processing avoids memory issues
- **Comprehensive Tracking**: Full timestamp tracking throughout pipeline
- **Fault Tolerance**: Resumable processing with partial completion support
- **Real-time Feedback**: Immediate partial results during processing

## 📖 Usage

**Main Entry Point:**
```bash
python main.py helm --benchmark lite
```

**Direct Script Usage:**
```bash
# Optimized processing
python scripts/optimized_helm_processor.py --benchmark lite --chunk-size 100

# Legacy processing  
python scripts/incremental_upload.py --benchmarks lite mmlu classic

# Statistics generation
python scripts/generate_comprehensive_stats.py --input-repo evaleval/every_eval_ever
```

For detailed implementation details, see the architecture documentation files listed above.
