"""
Data manager module for scraping and processing League of Legends data.
"""

import re
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List, Dict, Any
from config import METASRC_URL, GUIDE_URLS, CHAMPION_MAP


def get_champion_stats() -> List[Dict[str, Any]]:
    """
    Scrape champion statistics from MetaSrc.
    
    Returns:
        List[Dict[str, Any]]: List of champion statistics dictionaries.
    """
    # Set user agent to avoid being blocked
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        # Fetch the webpage
        response = requests.get(METASRC_URL, headers=headers)
        response.raise_for_status()  # Raise exception for bad responses
        
        # Parse the HTML
        soup = BeautifulSoup(response.content, "html.parser") 
        table = soup.select_one('#table-scroll table')
        
        if not table:
            print("Warning: Table not found. Check if the website structure has changed.")
            return []
            
        rows = table.find_all('tr')
        champions = []
        
        for row in rows:
            cells = row.find_all('td')
            data = [cell.get_text(strip=True) for cell in cells]
            if data:  # Skip empty lists
                # Fix for duplicate champion names
                champion_name = data[0]
                # Check if the name is duplicated (same text repeated)
                if len(champion_name) % 2 == 0:
                    half_length = len(champion_name) // 2
                    first_half = champion_name[:half_length]
                    second_half = champion_name[half_length:]
                    # If both halves are identical, use just one
                    if first_half == second_half:
                        champion_name = first_half
                
                champion = {
                    "Championname": champion_name,
                    "Lane": data[1],
                    "Tier": data[2],
                    "Score": data[3],
                    "Trend": data[4],
                    "Winrate": data[5],
                    "Rolerate": data[6],
                    "PickRate": data[7],
                    "BanRate": data[8],
                    "KDA": data[9]
                }
                champions.append(champion)
        
        print(f"Successfully scraped stats for {len(champions)} champions")
        return champions
        
    except Exception as e:
        print(f"Error scraping champion stats: {e}")
        return []


def create_stat_documents(champions_data: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert champion stats to Document objects for the vector store.
    
    Args:
        champions_data: List of champion statistics dictionaries.
        
    Returns:
        List[Document]: List of Document objects.
    """
    documents = []
    
    for champ in champions_data:
        text = f"Champion: {champ['Championname']}\n" + \
               f"Lane: {champ['Lane']}\n" + \
               f"Tier: {champ['Tier']}\n" + \
               f"Score: {champ['Score']}\n" + \
               f"Trend: {champ['Trend']}\n" + \
               f"Winrate: {champ['Winrate']}\n" + \
               f"Role rate: {champ['Rolerate']}\n" + \
               f"Pick rate: {champ['PickRate']}\n" + \
               f"Ban rate: {champ['BanRate']}\n" + \
               f"KDA: {champ['KDA']}\n\n"

        # Add interpretation of tier
        if "God" in champ["Tier"] or "S+" in champ["Tier"]:
            text += "This champion is the best and strongest pick for this lane in the current meta.\n"
        elif "S" in champ["Tier"] or "A+" in champ["Tier"]:
            text += "This champion is a strong pick for this lane in the current meta.\n"
        elif "A" in champ["Tier"] or "B+" in champ["Tier"]:
            text += "This champion is a good pick for this lane in the current meta.\n"
        elif "B" in champ["Tier"] or "C+" in champ["Tier"]:
            text += "This champion is an average pick for this lane in the current meta.\n"
        else:
            text += "This champion is a below average pick for this lane in the current meta.\n"
        
        documents.append(Document(
            page_content=text,
            metadata={
                "champion": champ['Championname'],
                "lane": champ['Lane'],
                "tier": champ['Tier'],
                "content_type": "statistics"
            }
        ))
    
    return documents


def extract_guide_id(url: str) -> str:
    """
    Extract guide ID from URL.
    
    Args:
        url: Guide URL.
        
    Returns:
        str: Guide ID.
    """
    match = re.search(r'-(\d+)$', url)
    if match:
        return match.group(1)
    return "unknown"


def load_champion_guides() -> List[Document]:
    """
    Load and process champion guides from web sources.
    
    Returns:
        List[Document]: List of Document objects.
    """
    all_chunks = []
    
    for url in GUIDE_URLS:
        try:
            print(f"Loading guide: {url}")
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            # Determine champion by URL
            guide_id = extract_guide_id(url)
            champion_name = CHAMPION_MAP.get(guide_id, "Unknown")
            
            # Add champion metadata
            for doc in documents:
                doc.metadata["champion"] = champion_name
                doc.metadata["guide_url"] = url
                doc.metadata["content_type"] = "guide"
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n## ", "\n### ", "\n\n", "\n", ".", "?", "!"]
            )
            chunks = text_splitter.split_documents(documents)
            
            # Analyze chunk content to determine sections
            for chunk in chunks:
                text_lower = chunk.page_content.lower()
                
                # Determine guide sections
                if any(term in text_lower for term in ["item", "build"]):
                    chunk.metadata["section"] = "items"
                elif any(term in text_lower for term in ["rune", "runes"]):
                    chunk.metadata["section"] = "runes"
                elif any(term in text_lower for term in ["skill", "ability", "abilities"]):
                    chunk.metadata["section"] = "skills"
                elif any(term in text_lower for term in ["matchup", "match-up", "versus", "vs", "counter"]):
                    chunk.metadata["section"] = "matchups"
                else:
                    chunk.metadata["section"] = "general"
            
            print(f"Created {len(chunks)} chunks for {champion_name}'s guide")
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"Error loading {url}: {e}")
    
    return all_chunks


def collect_data() -> List[Document]:
    """
    Collect all data and return it as a list of Documents.
    
    Returns:
        List[Document]: Combined list of all Document objects.
    """
    # Get champion statistics
    print("Fetching champion statistics...")
    champion_stats = get_champion_stats()
    stat_documents = create_stat_documents(champion_stats)
    print(f"Created {len(stat_documents)} documents from champion statistics")
    
    # Get champion guides
    guide_documents = load_champion_guides()
    
    # Combine all documents
    all_documents = stat_documents + guide_documents
    print(f"Total documents/chunks: {len(all_documents)}")
    
    return all_documents