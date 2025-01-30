"""Metadata extraction and management utilities.

This module handles:
1. Extracting unique identifiers from raw data
2. Saving/Loading metadata lists
3. Maintaining data dictionaries
"""

import pandas as pd
import logging
from pathlib import Path
from config import DATA_PARAMS
import json

logger = logging.getLogger(__name__)

def extract_metadata(raw_data_path: str) -> None:
    """Extract and save metadata lists from raw data."""
    logger.info("Extracting metadata from raw data...")
    
    # Create output directory if needed
    metadata_dir = Path(DATA_PARAMS['raw_data_path']) 
    
    try:
        # Load raw data
        df = pd.read_csv(
            f"{raw_data_path}/raw_data.csv",
            usecols=['FACTORNAME', 'COMPANYNAME', 'EXCHANGETICKER'],
            dtype='string'
        )
        logger.info(f"Loaded raw data with shape: {df.shape}")
        
        # Extract unique values
        factors = df['FACTORNAME'].dropna().unique().tolist()
        companies = df['COMPANYNAME'].dropna().unique().tolist() 
        exchanges = df['EXCHANGETICKER'].str.split(':', expand=True)[0].dropna().unique().tolist()
        
        # Save lists
        save_list(factors, metadata_dir / 'factor_list.txt')
        save_list(companies, metadata_dir / 'company_list.txt')
        save_list(exchanges, metadata_dir / 'exchange_list.txt')
        
        logger.info(f"Saved metadata for {len(factors)} factors, {len(companies)} companies, {len(exchanges)} exchanges")
        
    except Exception as e:
        logger.error(f"Metadata extraction failed: {str(e)}")
        raise

def save_list(items: list, file_path: Path) -> None:
    """Save a list to a text file with one item per line."""
    with open(file_path, 'w') as f:
        for item in sorted(items):
            f.write(f"{item}\n")
            
def load_list(file_path: Path) -> list:
    """Load a list from a text file."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extract_metadata(DATA_PARAMS['raw_data_path']) 