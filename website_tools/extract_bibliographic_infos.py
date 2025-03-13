import re
# New function to extract bibliographic information from academic content
def extract_bibliographic_info(content, doc_type, is_academic):
    """
    Extract bibliographic information from academic content if available
    
    Args:
        content (str): The document content
        doc_type (str): The document type (pdf, docx, etc)
        is_academic (bool): Whether the document is from an academic source
        
    Returns:
        dict: Dictionary with bibliographic information if found
    """
    if not is_academic:
        return None
    
    biblio_info = {
        "title": None,
        "authors": None,
        "publication": None,
        "year": None,
        "doi": None
    }
    
    # Try to extract title - often at the beginning or in a specific format
    title_patterns = [
        r'(?i)(?:title|paper):\s*([^\n\.]+)',
        r'^\s*#\s+([^\n\.]+)',  # Markdown title
        r'^\s*([A-Z][^\.]+?)\s*\n',  # First sentence in all caps or title case
        r'(?<=abstract)\s*([^\n\.]+)'  # Title before abstract
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, content[:500])
        if match:
            title = match.group(1).strip()
            if len(title) > 10 and len(title) < 200:  # Reasonable title length
                biblio_info["title"] = title
                break
    
    # Try to extract authors
    author_patterns = [
        r'(?i)(?:author|by)(?:s)?:\s*([^\n\.]+)',
        r'(?i)([A-Za-z\s\.,]+(?:et al\.)?)\s*\([0-9]{4}\)',  # Author(s) followed by year
        r'(?i)([A-Za-z\s\.,]+)(?:,|\sand\s)(?:[A-Za-z\s\.,]+)(?:\.|,)'  # Multiple authors separated by commas or 'and'
    ]
    
    for pattern in author_patterns:
        match = re.search(pattern, content[:1000])
        if match:
            authors = match.group(1).strip()
            if 3 < len(authors) < 200:  # Reasonable author string length
                biblio_info["authors"] = authors
                break
    
    # Try to extract year
    year_match = re.search(r'(?:19|20)[0-9]{2}', content[:1000])
    if year_match:
        biblio_info["year"] = year_match.group(0)
    
    # Try to extract DOI
    doi_match = re.search(r'(?i)(?:doi|https?://doi\.org/)[\s:]*(10\.[0-9]{4,}[^\s\n\.]+)', content)
    if doi_match:
        biblio_info["doi"] = doi_match.group(1)
    
    # Try to extract publication name
    pub_patterns = [
        r'(?i)(?:journal|conference|proceedings)[\s:]+([^\n\.]+)',
        r'(?i)in\s+([^,\.\n]+?),\s+(?:vol|volume|pp|pages)'
    ]
    
    for pattern in pub_patterns:
        match = re.search(pattern, content[:2000])
        if match:
            publication = match.group(1).strip()
            if 5 < len(publication) < 200:  # Reasonable publication name length
                biblio_info["publication"] = publication
                break
    
    # Only return if we found at least some information
    if any(value for value in biblio_info.values()):
        return biblio_info
    return None