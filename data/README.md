# LoL RAG Assistant Data Directory

This directory contains data files used by the LoL RAG Assistant.

## Directory Structure

- `lol_combined_vector_store/`: FAISS vector store for champion data (will be created when you run `scripts/build_vectorstore.py`)
- `models/`: Directory for cached model files (if needed)
- `evaluation/`: Evaluation results and visualizations (created by `scripts/evaluate_system.py`)

## How to Build the Vector Store

To build the vector store, run:

```bash
python scripts/build_vectorstore.py
```

This will scrape champion data from the web and build a FAISS vector store.

## Data Sources

The data is collected from the following sources:

1. Champion statistics: [MetaSrc](https://www.metasrc.com/lol/stats)
2. Champion guides: [MobaFire](https://www.mobafire.com/)

## Data Files

After running the data collection scripts, the following files will be created:

- `champion_stats.csv`: Champion statistics in CSV format
- `champion_stats.json`: Champion statistics in JSON format
- `guide_metadata.csv`: Metadata about champion guides
- `guide_chunks.json`: Chunked guide content for reference

## Ground Truth Data

For evaluation purposes, a ground truth dataset is defined in `config.py` for the following lanes:

- MID
- TOP
- ADC
- JUNGLE
- SUPPORT

Each lane has a list of champion names that are considered best picks according to expert knowledge.