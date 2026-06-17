import os
from typing import Annotated
from langchain_core.tools import tool
import wikipedia
from ddgs import DDGS

#1. Define the wikipedia search tool
@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for a query and return a summary."""
    try:
        summary = wikipedia.summary(query, sentences=3)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Your query is ambiguous. Here are some options: {e.options}"
    except wikipedia.exceptions.PageError:
        return "No page found for your query."
    except Exception as e:
        return f"An error occurred: {str(e)}"
    
#2. Define the DuckDuckGo search tool
@tool
def web_search(query: str, max_results: int = 3) -> str:
    """
    Search the live internet using DuckDuckGo. 
    Use this tool for current events, recent news, stock prices, or information after 2024.
    """
    try:
        # Initialize the DuckDuckGo search client
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            
        if not results:
            return f"No live web results found for '{query}'."
            
        # Format the snippet data into a clean text block for the LLM
        formatted_results = []
        for res in results:
            formatted_results.append(f"Title: {res['title']}\nURL: {res['href']}\nSnippet: {res['body']}\n---")
            
        return "\n".join(formatted_results)
    except Exception as e:
        return f"An error occurred during the web search: {str(e)}"
    
# tool to save the research findings to local files in text with json format for structured data. This is a simple implementation, in production you might want to use a database or cloud storage.
@tool
def save_research_findings(question: str, findings: list[str], sources: list[str]) -> str:
    """Save the research findings and sources to a local file."""
    try:
        filename = f"research_{question.replace(' ', '_')}.txt"
        with open(filename, "w") as f:
            f.write(f"Research Question: {question}\n\n")
            f.write("Findings:\n")
            for idx, finding in enumerate(findings, 1):
                f.write(f"{idx}. {finding}\n")
            f.write("\nSources:\n")
            for idx, source in enumerate(sources, 1):
                f.write(f"{idx}. {source}\n")
        return f"Research findings saved to {filename}"
    except Exception as e:
        return f"An error occurred while saving research findings: {str(e)}"