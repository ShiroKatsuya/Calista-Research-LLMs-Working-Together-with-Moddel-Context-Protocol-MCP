from websitetools import search_and_load, print_formatted_results

def test_academic_search():
    """Test the search and load function with academic content"""
    
    # We're specifically requesting academic content about language learning
    print("\n" + "=" * 80)
    print("Testing academic search for language learning research...")
    print("=" * 80)
    
    # Search with explicit academic terms
    results = search_and_load("research papers on language learning games educational benefits")
    
    # Print results with type info
    print(f"\nFound {len(results)} documents")
    
    # Summary of document types
    pdf_count = sum(1 for doc in results if doc.metadata.get('document_type') == 'pdf')
    doc_count = sum(1 for doc in results if doc.metadata.get('document_type') in ['doc', 'docx'])
    academic_count = sum(1 for doc in results if doc.metadata.get('is_academic', False))
    
    print(f"Document types: {pdf_count} PDFs, {doc_count} Word documents")
    print(f"Academic sources: {academic_count} out of {len(results)}")
    
    # Print detailed results
    print_formatted_results(results)
    
if __name__ == "__main__":
    test_academic_search() 