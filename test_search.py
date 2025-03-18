from websitetools import search_and_load
import os

def main():
    # Set API credentials
    os.environ["GOOGLE_CSE_ID"] = "5650b4d81ac344bd8"
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBEl28l8XlkhYWxCkGamWLAWW2Jfc3SJR0"
    
    # Test query
    query = "who is the current president of Indonesia 2025"
    print(f"Testing search for: {query}")
    
    # Perform search
    results = search_and_load(query, max_results=3)
    
    # Print results
    if results:
        print(f"Found {len(results)} results")
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Content preview: {doc.page_content[:200]}...")
    else:
        print("No results found")

if __name__ == "__main__":
    main() 