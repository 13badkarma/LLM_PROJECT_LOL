"""
Utility functions for the LoL RAG Assistant.
"""

import re
from typing import List, Dict, Tuple, Any

from config import GROUND_TRUTH


def extract_champions(response: str) -> List[str]:
    """
    Extract champion names from a response.
    
    Args:
        response: The response text.
        
    Returns:
        List[str]: List of champion names.
    """
    # Combine all possible champions from the ground truth
    all_possible_champions = []
    for champions in GROUND_TRUTH.values():
        for champion in champions:
            if champion not in all_possible_champions:
                all_possible_champions.append(champion)
    
    found_champions = []
    
    # Convert response to lowercase for case-insensitive matching
    response_lower = response.lower()
    
    for champion in all_possible_champions:
        # Simple string containment check
        if champion.lower() in response_lower:
            found_champions.append(champion)
    
    return found_champions


def calculate_accuracy(extracted_champions: List[str], ground_truth: List[str]) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        extracted_champions: List of extracted champion names.
        ground_truth: List of ground truth champion names.
        
    Returns:
        Tuple[float, float, float]: Precision, recall, and F1 score.
    """
    # Convert to lowercase for case-insensitive matching
    extracted_lower = [champ.lower() for champ in extracted_champions]
    ground_truth_lower = [champ.lower() for champ in ground_truth]
    
    true_positives = sum(1 for champ in extracted_lower if champ in ground_truth_lower)
    false_positives = len(extracted_lower) - true_positives
    false_negatives = sum(1 for champ in ground_truth_lower if champ not in extracted_lower)
    
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1


def get_true_positive_champions(extracted_champions: List[str], ground_truth: List[str]) -> List[str]:
    """
    Return the list of correctly identified champions.
    
    Args:
        extracted_champions: List of extracted champion names.
        ground_truth: List of ground truth champion names.
        
    Returns:
        List[str]: List of correctly identified champion names.
    """
    extracted_lower = [champ.lower() for champ in extracted_champions]
    
    true_positives = []
    for idx, champ_lower in enumerate(extracted_lower):
        if champ_lower in [gt.lower() for gt in ground_truth]:
            true_positives.append(extracted_champions[idx])
            
    return true_positives


def format_metadata_for_display(metadata: Dict[str, Any]) -> str:
    """
    Format document metadata for display.
    
    Args:
        metadata: Document metadata.
        
    Returns:
        str: Formatted metadata.
    """
    formatted = []
    for key, value in metadata.items():
        # Skip lengthy fields like full URLs
        if key == "guide_url" and isinstance(value, str) and len(value) > 50:
            value = value.split("/")[-1]
        formatted.append(f"{key.capitalize()}: {value}")
    
    return " | ".join(formatted)


def format_sources_for_display(sources: List[Any]) -> str:
    """
    Format source documents for display.
    
    Args:
        sources: List of source documents.
        
    Returns:
        str: Formatted sources.
    """
    if not sources:
        return "No sources available."
    
    formatted = []
    for i, doc in enumerate(sources):
        # Extract a short snippet from the content
        snippet = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        
        # Format the metadata
        meta_str = format_metadata_for_display(doc.metadata)
        
        # Combine into a source entry
        formatted.append(f"Source {i+1}:\n{snippet}\n{meta_str}\n")
    
    return "\n".join(formatted)