"""
RAG module for setting up and managing the retrieval-augmented generation pipeline.
"""

import os
from typing import Dict, Any, List
from langchain.vectorstores import FAISS
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document

from src.llm import llm
from src.embeddings import embedding_model
from src.data_manager import collect_data
from config import VECTOR_STORE_PATH, RAG_PROMPT_TEMPLATE, RETRIEVER_TOP_K


def build_vector_store(documents: List[Document] = None, force_rebuild: bool = False) -> FAISS:
    """
    Build or load the vector store.
    
    Args:
        documents: List of documents to build the vector store from. If None, will be collected.
        force_rebuild: If True, rebuild the vector store even if it exists.
    
    Returns:
        FAISS: The vector store.
    """
    # Check if vector store already exists
    if os.path.exists(VECTOR_STORE_PATH) and not force_rebuild:
        print(f"Loading existing vector store from {VECTOR_STORE_PATH}")
        vector_store = FAISS.load_local(str(VECTOR_STORE_PATH), embedding_model, allow_dangerous_deserialization=True)
    else:
        print("Building new vector store...")
        # Collect data if not provided
        if documents is None:
            documents = collect_data()
        
        # Create vector store
        vector_store = FAISS.from_documents(documents, embedding_model)
        
        # Save vector store
        os.makedirs(VECTOR_STORE_PATH.parent, exist_ok=True)
        vector_store.save_local(str(VECTOR_STORE_PATH))
        print(f"Vector store saved to {VECTOR_STORE_PATH}")
    
    return vector_store


def setup_rag_pipeline(vector_store: FAISS = None) -> RetrievalQA:
    """
    Set up the RAG pipeline.
    
    Args:
        vector_store: The vector store to use. If None, it will be loaded or built.
        
    Returns:
        RetrievalQA: The RAG pipeline.
    """
    # Load or build vector store if not provided
    if vector_store is None:
        vector_store = build_vector_store()
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_TOP_K}
    )
    
    # Set up the prompt template
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    # Create the RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return rag_chain


def get_response(query: str, rag_chain: RetrievalQA = None) -> Dict[str, Any]:
    """
    Get a response from the RAG pipeline.
    
    Args:
        query: The user's query.
        rag_chain: The RAG pipeline. If None, it will be created.
        
    Returns:
        Dict[str, Any]: A dictionary with the result and source documents.
    """
    # Set up RAG pipeline if not provided
    if rag_chain is None:
        rag_chain = setup_rag_pipeline()
    
    # Get response
    response = rag_chain.invoke({"query": query})
    
    return {
        "result": response["result"],
        "sources": response.get("source_documents", [])
    }