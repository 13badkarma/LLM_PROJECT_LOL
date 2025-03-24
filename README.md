# LoL RAG Assistant

A Retrieval-Augmented Generation (RAG) powered AI assistant for League of Legends players. This application provides accurate champion recommendations and insights using a combination of LLM technology with specialized knowledge of the current meta.

## Features

- 🎮 **Champion Recommendations**: Get the best champions for each lane based on current meta
- 📊 **Meta Analysis**: Up-to-date statistics on champion win rates, pick rates, and performance
- 🧠 **RAG Architecture**: Combines the power of LLMs with grounded knowledge retrieval
- 🔍 **Precision Focused**: Delivers factual information rather than hallucinations

## Project Structure


```
LLM_PROJECT_LOL/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py                       # FastAPI application
├── config.py                    # Configuration settings
├── data/                        # Data storage directory
│   └── README.md                # Instructions for data files
├── notebooks/                   # Original notebooks (reference only)
│   └── llm-project.ipynb        # Original implementation
├── scripts/                     # Utility scripts
│   ├── scrape_data.py           # Web scraping script
│   ├── build_vectorstore.py     # Vector store builder
│   └── evaluate_system.py       # Evaluation script
├── src/                         # Source code
│   ├── __init__.py
│   ├── data_manager.py          # Data collection and management
│   ├── embeddings.py            # Embedding model setup
│   ├── llm.py                   # LLM configuration
│   ├── rag.py                   # RAG pipeline
│   └── utils.py                 # Utility functions
└── static/                      # Static files for web interface
    ├── index.html               # Simple web interface
    ├── style.css                # CSS styles
    └── script.js                # JavaScript for interface
```




## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/lol-rag-assistant.git
   cd LLM_PROJECT_LOL
   ```
2. **Create a virtual environment**:
   ```python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Create .env file (optional)**:
```cat > .env << EOF
LLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
USE_GPU=true
ENABLE_API=true
API_HOST=0.0.0.0
API_PORT=8000
EOF
```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Prepare the vector database**:
   ```bash
   python scripts/build_vectorstore.py
   ```

6. **Start the application**:
   ```bash
   python app.py
   ```

7. **Access the web interface**:
   Open your browser and go to `http://localhost:8000`

## System Architecture

This project implements a RAG (Retrieval-Augmented Generation) system to ground LLM responses in factual information about League of Legends. The system:

1. **Retrieves** relevant information about champions from a vector database
2. **Augments** the language model's prompt with this contextual information
3. **Generates** accurate and informative responses about the current meta

The vector database is built from:
- Champion statistics scraped from MetaSrc
- Champion guides from MobaFire
- Lane-specific meta data

## Requirements

- Python 3.10+
- PyTorch
- Transformers (Hugging Face)
- FAISS for vector storage
- FastAPI for the web interface

## Deployment Options

### Local Deployment

The system is designed to run on a machine with a GPU for optimal performance, but can also run on CPU-only systems (with slower inference times).

### Cloud Deployment

The application can be deployed to:
- AWS EC2 (g4dn instances recommended)
- Google Cloud (with GPU support)
- Azure (with GPU support)
- Any VPS with sufficient RAM and optional GPU support

## Performance

Our RAG system significantly outperforms a standalone LLM:

- **Precision**: 1.00 vs 0.70 (+42.9%)
- **Recall**: 0.75 vs 0.45 (+66.7%)
- **F1 Score**: 0.83 vs 0.49 (+68.2%)


## License

This project is licensed under the MIT License - see the LICENSE file for details.