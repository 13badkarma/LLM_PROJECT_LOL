#!/usr/bin/env python3
"""
Main application file for the LoL RAG Assistant.
"""

import os
import json
from typing import Dict, List, Optional
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel

from src.rag import setup_rag_pipeline, get_response
from src.utils import format_sources_for_display
from config import API_HOST, API_PORT, ENABLE_API


# Initialize RAG pipeline at startup
print("Initializing LoL RAG Assistant...")
rag_chain = setup_rag_pipeline()
print("RAG pipeline initialized successfully!")


# Initialize FastAPI app
app = FastAPI(
    title="LoL RAG Assistant",
    description="Retrieval-Augmented Generation assistant for League of Legends players",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="static")


class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    show_sources: Optional[bool] = False


class QueryResponse(BaseModel):
    """Query response model."""
    result: str
    sources: Optional[List[Dict]] = None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the index page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/query", response_model=QueryResponse)
async def query(query_request: QueryRequest):
    """
    Process a query and return the result.
    
    Args:
        query_request: The query request.
        
    Returns:
        QueryResponse: The query response.
    """
    if not query_request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Get response from RAG pipeline
        result = get_response(query_request.query, rag_chain)
        
        # Process response
        response = {
            "result": result["result"],
        }
        
        # Include sources if requested
        if query_request.show_sources and result.get("sources"):
            response["sources"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["sources"]
            ]
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/query", response_class=HTMLResponse)
async def web_query(request: Request, query: str = Form(...), show_sources: bool = Form(False)):
    """
    Process a query from the web interface and return HTML response.
    
    Args:
        request: The request object.
        query: The query string.
        show_sources: Whether to show sources.
        
    Returns:
        HTMLResponse: The HTML response.
    """
    if not query.strip():
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "error": "Query cannot be empty"
            }
        )
    
    try:
        # Get response from RAG pipeline
        result = get_response(query, rag_chain)
        
        # Format response for display
        context = {
            "request": request,
            "query": query,
            "result": result["result"].replace("\n", "<br>"),
            "show_sources": show_sources,
        }
        
        # Include sources if requested
        if show_sources and result.get("sources"):
            context["sources"] = format_sources_for_display(result["sources"])
        
        return templates.TemplateResponse("index.html", context)
    
    except Exception as e:
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "query": query,
                "error": f"Error processing query: {str(e)}"
            }
        )


if __name__ == "__main__" and ENABLE_API:
    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=True)