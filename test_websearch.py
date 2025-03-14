#!/usr/bin/env python3
"""
Web Search Test Utility

This script demonstrates how to use the web search functionality
from the websitetools module directly.

Usage:
  python test_websearch.py "your search query"
"""

import sys
from websitetools import runs

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_websearch.py \"your search query\"")
        print("Example: python test_websearch.py \"latest AI research advances\"")
        return
    
    # Get the search query from command line arguments
    query = " ".join(sys.argv[1:])
    print(f"\n[System] Performing web search for: {query}\n")
    
    # Execute the web search using the runs function
    try:
        runs(query)
        print("\n[System] Web search completed successfully.\n")
    except Exception as e:
        print(f"\n[ERROR] Web search failed: {str(e)}\n")

if __name__ == "__main__":
    main() 