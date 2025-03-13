from urllib.parse import unquote
# Function to detect if URL is likely from an academic source
def is_academic_source(url):
    """
    Check if a URL is likely from an academic or scholarly source
    
    Args:
        url (str): The URL to check
        
    Returns:
        bool: True if likely academic, False otherwise
    """
    academic_domains = [
        '.edu', '.ac.', '.gov', 'scholar.google', 'researchgate.net', 
        'academia.edu', 'jstor.org', 'sciencedirect.com', 'springer.com',
        'ieee.org', 'ncbi.nlm.nih.gov', 'pubmed', 'arxiv.org',
        'ssrn.com', 'nature.com', 'science.org', 'wiley.com', 'tandfonline.com',
        'sagepub.com', 'elsevier.com', 'acm.org', 'apa.org'
    ]
    
    academic_keywords = [
        'journal', 'research', 'proceedings', 'conference', 'thesis',
        'dissertation', 'paper', 'study', 'article', 'peer-review'
    ]
    
    # Check domain
    if any(domain in url.lower() for domain in academic_domains):
        return True
        
    # Check for PDF links with academic keywords
    if url.lower().endswith('.pdf'):
        decoded_url = unquote(url.lower())
        if any(keyword in decoded_url for keyword in academic_keywords):
            return True
            
    return False