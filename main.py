#!/usr/bin/env python3
"""
Every Eval Ever - Main Entry Point

A unified interface for processing evaluation data from multiple sources.
Currently supports HELM with plans to expand to other evaluation frameworks.

Usage:
    python main.py helm --benchmark lite
    python main.py bigbench --dataset gsm8k  # Future
    python main.py mmlu --split dev          # Future
"""

import argparse
import sys
import importlib.util
from pathlib import Path


def load_source_processor(source_name: str):
    """Dynamically load the processor for a given data source."""
    processor_path = Path(f"src/sources/{source_name}/processor.py")
    
    if not processor_path.exists():
        available_sources = [p.name for p in Path("src/sources").iterdir() if p.is_dir()]
        raise ValueError(
            f"Data source '{source_name}' not found.\n"
            f"Available sources: {', '.join(available_sources)}"
        )
    
    # Import the module using the proper module path for multiprocessing compatibility
    module_name = f"src.sources.{source_name}.processor"
    spec = importlib.util.spec_from_file_location(module_name, processor_path)
    module = importlib.util.module_from_spec(spec)
    
    # Register the module in sys.modules so multiprocessing can find it
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    return module


def main():
    parser = argparse.ArgumentParser(
        description="Process evaluation data from multiple sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py helm --benchmark lite
  python main.py helm --benchmark mmlu --max-workers 4
  python main.py helm --benchmark classic --overwrite
        """
    )
    
    # Get available data sources
    sources_dir = Path("src/sources")
    if sources_dir.exists():
        available_sources = [p.name for p in sources_dir.iterdir() if p.is_dir()]
    else:
        available_sources = []
    
    parser.add_argument(
        "source",
        choices=available_sources,
        help="Data source to process"
    )
    
    # Parse just the source first to determine which processor to load
    if len(sys.argv) < 2:
        parser.print_help()
        return 1
    
    source_name = sys.argv[1]
    
    if source_name in ['-h', '--help']:
        parser.print_help()
        return 0
    
    if source_name not in available_sources:
        print(f"Error: Unknown data source '{source_name}'")
        print(f"Available sources: {', '.join(available_sources)}")
        return 1
    
    try:
        # Load the source-specific processor
        processor_module = load_source_processor(source_name)
        
        # Remove the source name from sys.argv and run the processor
        sys.argv = [f"{source_name}_processor"] + sys.argv[2:]
        
        # Call the processor's main function
        if hasattr(processor_module, 'main'):
            return processor_module.main()
        else:
            print(f"Error: {source_name} processor doesn't have a main() function")
            return 1
            
    except Exception as e:
        print(f"Error loading {source_name} processor: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
