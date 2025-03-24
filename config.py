"""
Configuration module for LoL RAG Assistant.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
VECTOR_STORE_PATH = DATA_DIR / 'lol_combined_vector_store'
MODELS_DIR = DATA_DIR / 'models'

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
VECTOR_STORE_PATH.parent.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# LLM Configuration
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# API Configuration
ENABLE_API = os.getenv("ENABLE_API", "true").lower() == "true"
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Data sources
METASRC_URL = "https://www.metasrc.com/lol/stats"
GUIDE_URLS = [
    "https://www.mobafire.com/league-of-legends/build/25-05-the-bible-of-jungling-shaco-white-crows-sexy-guide-327054",
    "https://www.mobafire.com/league-of-legends/build/25-s1-5-fundamentals-of-aatrox-diamond-guide-remastered-632525",
    "https://www.mobafire.com/league-of-legends/build/25-05-shoks-rank-1-challenger-ahri-guide-635065"
]

# Champion mapping for guides (ID to champion name)
CHAMPION_MAP = {
    "327054": "Shaco",
    "632525": "Aatrox",
    "635065": "Ahri"
}

# RAG Configuration
RETRIEVER_TOP_K = 5
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
DEVICE = "cuda:0" if USE_GPU and os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
EMBEDDING_DEVICE = "cuda:1" if USE_GPU and os.getenv("CUDA_VISIBLE_DEVICES") and "," in os.getenv("CUDA_VISIBLE_DEVICES", "") else "cpu"

# Prompt template for RAG
RAG_PROMPT_TEMPLATE = """
Using the following information from League of Legends guides and champion statistics, answer the user's question.
Context:
{context}
Question: {question}
Please provide a detailed answer based only on the provided information.
If there's no information in the context to answer the question, please say so.
If providing statistical information, mention how strong the champion is in the current meta.
If providing guide information, indicate which section of the guide it's from.
"""

# Ground truth champions for evaluation
GROUND_TRUTH = {
    "MID": ["Ahri", "Mel", "Sylas", "Zed"],
    "TOP": ["Aatrox", "Darius", "Sett"],
    "ADC": ["Caitlyn", "Ezreal", "Jhin", "Jinx"],
    "JUNGLE": ["Darius", "Viego", "Lee Sin"],
    "SUPPORT": ["Karma", "Lulu", "Lux"]
}

# Queries for evaluation
EVAL_QUERIES = {
    "MID": "What are the best champions for MID?",
    "TOP": "What are the best champions for TOP?",
    "ADC": "What are the best champions for ADC?",
    "JUNGLE": "What are the best champions for JUNGLE?",
    "SUPPORT": "What are the best champions for SUPPORT?"
}