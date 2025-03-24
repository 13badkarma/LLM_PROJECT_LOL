#!/usr/bin/env python3
"""
Script to build the vector store from LoL data.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag import build_vector_store
from src.data_manager import collect_data


def main():
    """Build the vector store."""
    parser = argparse.ArgumentParser(description="Build vector store for LoL RAG Assistant")
    parser.add_argument("--force", action="store_true", help="Force rebuild the vector store")
    args = parser.parse_args()
    
    print("Building LoL RAG Assistant Vector Store")
    print("---------------------------------------")
    
    # Collect data
    documents = collect_data()
    
    # Build vector store
    vector_store = build_vector_store(documents, force_rebuild=args.force)
    
    print("\nVector store built successfully!")
    print(f"Number of documents: {vector_store.index.ntotal}")


if __name__ == "__main__":
    main()