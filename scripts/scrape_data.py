#!/usr/bin/env python3
"""
Script to scrape LoL champion data from the web.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_manager import get_champion_stats, load_champion_guides
from config import DATA_DIR


def main():
    """Scrape LoL champion data."""
    parser = argparse.ArgumentParser(description="Scrape LoL champion data from the web")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    output_dir = args.output if args.output else DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    print("Scraping LoL Champion Data")
    print("-------------------------")
    
    # Scrape champion stats
    print("\nScraping champion statistics...")
    champion_stats = get_champion_stats()
    
    if champion_stats:
        # Save to CSV
        stats_df = pd.DataFrame(champion_stats)
        stats_file = os.path.join(output_dir, 'champion_stats.csv')
        stats_df.to_csv(stats_file, index=False)
        print(f"Saved champion stats to {stats_file}")
        
        # Save to JSON for backup
        json_file = os.path.join(output_dir, 'champion_stats.json')
        with open(json_file, 'w') as f:
            json.dump(champion_stats, f, indent=2)
        print(f"Saved champion stats to {json_file}")
    else:
        print("Failed to scrape champion stats")
    
    # Scrape champion guides
    print("\nScraping champion guides...")
    guide_documents = load_champion_guides()
    
    if guide_documents:
        # Save guide metadata
        guide_meta = [
            {
                "champion": doc.metadata.get("champion", "Unknown"),
                "guide_url": doc.metadata.get("guide_url", "Unknown"),
                "section": doc.metadata.get("section", "Unknown"),
                "content_length": len(doc.page_content)
            }
            for doc in guide_documents
        ]
        meta_df = pd.DataFrame(guide_meta)
        meta_file = os.path.join(output_dir, 'guide_metadata.csv')
        meta_df.to_csv(meta_file, index=False)
        print(f"Saved guide metadata to {meta_file}")
        
        # Save guide chunks for reference
        chunks_file = os.path.join(output_dir, 'guide_chunks.json')
        guide_data = [
            {
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata
            }
            for doc in guide_documents
        ]
        with open(chunks_file, 'w') as f:
            json.dump(guide_data, f, indent=2)
        print(f"Saved guide chunks to {chunks_file}")
    else:
        print("Failed to scrape champion guides")
    
    print("\nData scraping complete!")


if __name__ == "__main__":
    main()