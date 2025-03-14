#!/usr/bin/env python3
"""
Web Search Debug Utility

This script helps diagnose issues with the web search functionality
from the websitetools module.

Usage:
  python debug_websearch.py "your search query"
"""

import sys
import traceback
from websitetools import runs, search_and_load, print_formatted_results

def debug_search(query):
    """
    Run a debug search with extensive error reporting.
    
    Args:
        query: The search query to use
    """
    print(f"\n===== DEBUGGING WEB SEARCH =====")
    print(f"Query: '{query}'")
    
    # Test method 1: Using runs function
    print("\n[TEST 1] Using runs() function:")
    try:
        print("Running: runs(query)")
        runs(query)
        print("[RESULT] runs() function completed successfully.")
    except Exception as e:
        print(f"[ERROR] runs() function failed: {str(e)}")
        print("\nStack trace:")
        traceback.print_exc()
    
    # Test method 2: Direct search_and_load call
    print("\n[TEST 2] Using search_and_load() function directly:")
    try:
        print("Running: results = search_and_load(query)")
        results = search_and_load(query)
        print(f"[RESULT] search_and_load() returned {len(results) if results else 0} results.")
        
        if results:
            print("\nPrinting formatted results:")
            print_formatted_results(results)
    except Exception as e:
        print(f"[ERROR] search_and_load() function failed: {str(e)}")
        print("\nStack trace:")
        traceback.print_exc()
    
    print("\n===== DEBUG COMPLETE =====")

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_websearch.py \"your search query\"")
        print("Example: python debug_websearch.py \"latest AI research advances\"")
        return
    
    # Get the search query from command line arguments
    query = " ".join(sys.argv[1:])
    debug_search(query)

if __name__ == "__main__":
    main() 