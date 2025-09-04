#!/usr/bin/env python3
"""
Show repository structure overview
"""

from pathlib import Path

def show_repo_structure():
    """Display the organized repository structure."""
    
    print("📁 Repository Structure Overview")
    print("=" * 50)
    
    # Show tests directory
    tests_dir = Path(__file__).parent
    if tests_dir.name == "tests":
        print(f"\n📁 {tests_dir}/ (Test directory)")
        for item in sorted(tests_dir.iterdir()):
            if item.is_file():
                size = item.stat().st_size
                if size > 1024:
                    size_str = f"({size/1024:.1f} KB)"
                else:
                    size_str = f"({size} bytes)"
                print(f"  📄 {item.name} {size_str}")
            elif item.is_dir() and not item.name.startswith('.'):
                print(f"  📁 {item.name}/")
                for sub_item in sorted(item.iterdir()):
                    if sub_item.is_file():
                        size = sub_item.stat().st_size
                        if size > 1024:
                            size_str = f"({size/1024:.1f} KB)"
                        else:
                            size_str = f"({size} bytes)"
                        print(f"    📄 {sub_item.name} {size_str}")
    
    # Show scripts directory
    scripts_dir = tests_dir.parent / "scripts"
    if scripts_dir.exists():
        print(f"\n📁 scripts/ (Main processing scripts)")
        for item in sorted(scripts_dir.iterdir()):
            if item.is_file() and item.suffix == '.py':
                print(f"  📄 {item.name}")
    
    print(f"\n📊 Key Files:")
    print(f"  🚀 scripts/optimized_helm_processor.py - Main cronjob-ready processor")
    print(f"  📈 scripts/generate_comprehensive_stats.py - Stats generation for scores repo")
    print(f"  🧪 tests/test_end_to_end.py - Complete pipeline validation")
    print(f"  📄 tests/examples/ - Sample parquet files for reference")
    
    print(f"\n📋 File Naming Convention:")
    print(f"  📁 Local processing: chunk_XXXX.parquet")
    print(f"  ☁️  HuggingFace upload: data-XXXXX.parquet")
    print(f"  📊 Summary stats: comprehensive_stats_YYYY-MM-DD.parquet")

if __name__ == "__main__":
    show_repo_structure()
