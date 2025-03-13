from bs4 import BeautifulSoup
def extract_main_content(html_content, url):

    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove common non-content elements
    for element in soup.find_all(['header', 'footer', 'nav', 'aside', 'script', 'style', 'noscript', 
                                 'iframe', 'form', 'button', 'svg']):
        element.decompose()
        
    # Remove common ad, navigation, and footer classes/IDs
    for selector in [
        '[class*="nav"]', '[class*="menu"]', '[class*="header"]', '[class*="footer"]',
        '[class*="banner"]', '[class*="ad-"]', '[class*="ads-"]', '[class*="widget"]',
        '[class*="sidebar"]', '[class*="comment"]', '[id*="nav"]', '[id*="menu"]',
        '[id*="header"]', '[id*="footer"]', '[id*="banner"]', '[id*="ad-"]', 
        '[id*="sidebar"]', '[id*="comment"]'
    ]:
        for element in soup.select(selector):
            element.decompose()
    

    main_content = None
    content_elements = soup.select('article, [class*="content"], [class*="post"], [class*="article"], main, #content, .content, .post, .article')
    
    if content_elements:
        # Use the largest content block (likely the main article)
        main_content = max(content_elements, key=lambda x: len(x.get_text(strip=True)))
    else:
        # Fallback: find div with the most paragraphs
        divs = soup.find_all('div')
        if divs:
            main_content = max(divs, key=lambda x: len(x.find_all('p')))
        else:
            main_content = soup.body
    
    if not main_content:
        return "Could not extract main content"
    
    # Extract text from main content, preserving only necessary HTML elements
    text = ""
    for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'blockquote']):
        element_text = element.get_text(strip=True)
        if element_text:
            if element.name.startswith('h'):
                text += f"\n### {element_text} ###\n"
            elif element.name == 'li':
                text += f"  • {element_text}\n"
            elif element.name == 'blockquote':
                text += f"> {element_text}\n"
            else:
                text += f"{element_text}\n\n"
    
    return text.strip()