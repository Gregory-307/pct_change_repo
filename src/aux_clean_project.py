"""Script to clean the project, keeping only core components."""

import shutil
from pathlib import Path

def clean_project():
    """Clean the project, keeping only source code, raw data, and documentation."""
    # Directories to completely remove
    dirs_to_remove = [
        'results',
        'models',
        'data/processed',
        'data/analysis',
        'best_model',
        'bin',
        'temp_backup'
    ]
    
    # Remove directories
    for dir_path in dirs_to_remove:
        dir_path = Path(dir_path)
        if dir_path.exists():
            print(f"Removing {dir_path}...")
            shutil.rmtree(dir_path)
    
    # Ensure core directories exist
    core_dirs = [
        'data/raw',
        'src'
    ]
    
    for dir_path in core_dirs:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

def main():
    print("Starting project cleanup...")
    print("\nThe following will be kept:")
    print("1. Source code (src/)")
    print("2. Raw data (data/raw/raw_data.csv)")
    print("3. README.md and requirements.txt")
    
    response = input("\nThis will delete all generated files and models. Continue? (y/n): ")
    if response.lower() == 'y':
        clean_project()
        print("\nCleanup complete! Project structure reset to core components.")
        print("\nNew structure:")
        print("- src/ (source code)")
        print("- data/")
        print("  - raw/ (raw_data.csv)")
        print("- README.md")
        print("- requirements.txt")
    else:
        print("\nCleanup cancelled.")

if __name__ == "__main__":
    main() 