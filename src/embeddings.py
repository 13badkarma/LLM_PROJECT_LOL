"""
Embeddings module for vector representations of text.
"""

import torch
from langchain.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE


def get_embedding_model():
    """
    Initialize and return the embedding model.
    
    Returns:
        HuggingFaceEmbeddings: The embedding model instance.
    """
    # Determine if we can use GPU for embeddings
    device = EMBEDDING_DEVICE if torch.cuda.is_available() else "cpu"
    
    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device}
    )
    
    print(f"Embedding model loaded on device: {device}")
    
    return embedding_model


# Create a singleton instance for reuse
embedding_model = get_embedding_model()