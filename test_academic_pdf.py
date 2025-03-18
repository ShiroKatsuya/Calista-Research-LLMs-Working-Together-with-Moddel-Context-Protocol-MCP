from websitetools import search_and_load, print_formatted_results

def test_academic_pdf_search():
    """Test the search and load function with academic PDF content"""
    
    print("\n" + "=" * 80)
    print("Testing academic PDF search for language learning and games...")
    print("=" * 80)
    
    # Search with explicit academic terms and PDF format
    results = search_and_load("PDF language learning educational games research filetype:pdf")
    
    # Print results with type info
    print(f"\nFound {len(results)} documents")
    
    # Summary of document types
    pdf_count = sum(1 for doc in results if doc.metadata.get('document_type') == 'pdf')
    academic_count = sum(1 for doc in results if doc.metadata.get('is_academic', False))
    
    print(f"Document types: {pdf_count} PDFs")
    print(f"Academic sources: {academic_count} out of {len(results)}")
    
    # For PDFs, print some additional information
    for i, doc in enumerate(results, 1):
        if doc.metadata.get('document_type') == 'pdf':
            print(f"\n[PDF {i}] Source: {doc.metadata.get('source')}")
            print(f"Academic: {'Yes' if doc.metadata.get('is_academic', False) else 'No'}")
            
            # Print a short preview of content (first 200 chars)
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"Content preview: {content_preview}")
    
    # Print detailed results
    print_formatted_results(results)
    
if __name__ == "__main__":
    test_academic_pdf_search() 