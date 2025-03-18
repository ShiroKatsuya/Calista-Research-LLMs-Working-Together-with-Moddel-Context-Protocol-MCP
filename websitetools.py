import os
from langchain_core.tools import Tool
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
import re
import textwrap
import urllib3
import requests
import os.path
from website_tools.query_google import query_googling
from website_tools.detect_documents_type import detect_document_type
from website_tools.extract_text_from_pdfs import extract_text_from_pdf
from website_tools.extract_text_from_docxs import extract_text_from_docx
from website_tools.is_academic_sources import is_academic_source
from website_tools.extract_main_contents import extract_main_content
from website_tools.score_paragraph_relevances import score_paragraph_relevance
from website_tools.extract_keywords_from_querys import extract_keywords_from_query
from website_tools.extract_keywords_from_texts import extract_keywords_from_text
from website_tools.generate_summarys import generate_summary
from website_tools.format_contents import format_content
from website_tools.extract_bibliographic_infos import extract_bibliographic_info
# import nltk
# try:
#     nltk.data.find('tokenizers/punkt')
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('punkt')
#     nltk.download('stopwords')
os.environ["GOOGLE_CSE_ID"]
google_api1 = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = google_api1
os.environ["USER_AGENT"]
def extract_text_from_document_url(url, query):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml,application/pdf,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'https://www.google.com/'  # Add a referrer to help with access
        }
        
        # Add proper timeout handling
        response = requests.get(url, headers=headers, verify=False, timeout=30, stream=True)
        
        # Check if response was successful
        if response.status_code != 200:
            # For 403 errors, try a different approach with a simplified request
            if response.status_code == 403:
                try:
                    # Try with a simpler set of headers that might bypass some restrictions
                    simple_headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36',
                    }
                    response = requests.get(url, headers=simple_headers, verify=False, timeout=30)
                    if response.status_code == 200:
                        print(f"Successfully accessed {url} with simplified headers")
                    else:
                        return f"[Error: HTTP status code {response.status_code}. The website may be blocking automated access]", "unknown", False
                except Exception as retry_error:
                    return f"[Error: HTTP status code {response.status_code}. The website may be blocking automated access]", "unknown", False
            else:
                return f"[Error: HTTP status code {response.status_code}]", "unknown", False
            
        # Add content length check to avoid processing extremely large files
        content_length = response.headers.get('Content-Length')
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
            return f"[Error: Document too large ({int(content_length) // (1024 * 1024)}MB)]", "unknown", False
        
        # Detect document type with better error handling
        try:
            doc_type = detect_document_type(url, response)
        except Exception as e:
            print(f"Error detecting document type for {url}: {e}")
            doc_type = "unknown"
        
        # Check if document type is supported
        if doc_type == "unknown":
            # Try to detect based on content-type header
            content_type = response.headers.get('Content-Type', '').lower()
            if 'pdf' in content_type:
                doc_type = 'pdf'
            elif 'word' in content_type or 'docx' in content_type:
                doc_type = 'docx'
            elif 'text/plain' in content_type:
                doc_type = 'txt'
            elif 'html' in content_type:
                doc_type = 'html'
        
        is_academic = is_academic_source(url)
        
        metadata = {
            "source": url,
            "document_type": doc_type,
            "is_academic": is_academic
        }
        
        # Process different document types with specific error handling
        if doc_type == 'pdf':
            try:
                content = extract_text_from_pdf(response.content)
                if not content or len(content.strip()) < 20:
                    return f"[Error: Could not extract text from PDF]", doc_type, is_academic
                return content, doc_type, is_academic
            except Exception as e:
                print(f"Error extracting PDF text from {url}: {e}")
                return f"[Error extracting PDF: {str(e)}]", doc_type, is_academic
            
        elif doc_type in ['docx', 'doc']:
            try:
                content = extract_text_from_docx(response.content)
                if not content or len(content.strip()) < 20:
                    return f"[Error: Could not extract text from DOCX]", doc_type, is_academic
                return content, doc_type, is_academic
            except Exception as e:
                print(f"Error extracting DOCX text from {url}: {e}")
                return f"[Error extracting DOCX: {str(e)}]", doc_type, is_academic
            
        elif doc_type == 'txt':
            # Use encoding detection for text files
            try:
                encoding = response.encoding or 'utf-8'
                return response.text, doc_type, is_academic
            except UnicodeDecodeError:
                # Try different encodings if the default fails
                try:
                    return response.content.decode('latin-1'), doc_type, is_academic
                except:
                    print(f"Error decoding text from {url}")
                    return f"[Error: Could not decode text content]", doc_type, is_academic
            
        elif doc_type == 'html':
            try:
                # First try with the main content extractor
                content = extract_main_content(response.text, url)
                
                # Check if content was successfully extracted
                if not content or len(content.strip()) < 50:
                    print(f"Main content extraction failed for {url}, using fallback method")
                    
                    # Fallback 1: Simple HTML parsing - extract text from common content areas
                    simple_content = ""
                    html_text = response.text.lower()
                    
                    # Look for common content containers
                    content_patterns = [
                        r'<article[^>]*>(.*?)</article>',
                        r'<main[^>]*>(.*?)</main>',
                        r'<div[^>]*class=["\']content["\'][^>]*>(.*?)</div>',
                        r'<div[^>]*class=["\']post["\'][^>]*>(.*?)</div>',
                        r'<div[^>]*id=["\']content["\'][^>]*>(.*?)</div>',
                        r'<div[^>]*id=["\']main["\'][^>]*>(.*?)</div>'
                    ]
                    
                    for pattern in content_patterns:
                        matches = re.findall(pattern, html_text, re.DOTALL | re.IGNORECASE)
                        if matches:
                            # Use the longest match as it's likely the main content
                            simple_content = max(matches, key=len)
                            break
                    
                    if simple_content and len(simple_content.strip()) > 100:
                        # Simple HTML tag stripping
                        simple_content = re.sub(r'<[^>]+>', ' ', simple_content)
                        simple_content = re.sub(r'\s{2,}', ' ', simple_content)
                        content = simple_content.strip()
                    else:
                        # Fallback 2: Just strip tags from the whole body
                        body_match = re.search(r'<body[^>]*>(.*?)</body>', html_text, re.DOTALL | re.IGNORECASE)
                        if body_match:
                            body_content = body_match.group(1)
                            # Remove scripts, styles, and other non-content elements
                            body_content = re.sub(r'<script[^>]*>.*?</script>', '', body_content, flags=re.DOTALL | re.IGNORECASE)
                            body_content = re.sub(r'<style[^>]*>.*?</style>', '', body_content, flags=re.DOTALL | re.IGNORECASE)
                            body_content = re.sub(r'<nav[^>]*>.*?</nav>', '', body_content, flags=re.DOTALL | re.IGNORECASE)
                            body_content = re.sub(r'<footer[^>]*>.*?</footer>', '', body_content, flags=re.DOTALL | re.IGNORECASE)
                            body_content = re.sub(r'<header[^>]*>.*?</header>', '', body_content, flags=re.DOTALL | re.IGNORECASE)
                            # Strip HTML tags
                            body_content = re.sub(r'<[^>]+>', ' ', body_content)
                            body_content = re.sub(r'\s{2,}', ' ', body_content)
                            content = body_content.strip()
                        else:
                            # Fallback 3: Use full HTML content as last resort
                            print(f"All extraction methods failed for {url}, using full HTML")
                            content = response.text
                
                if not content or len(content.strip()) < 50:
                    # If we still don't have good content, use the raw HTML
                    content = response.text
                    
                return content, doc_type, is_academic
                
            except Exception as e:
                print(f"Error extracting HTML content from {url}: {e}")
                # Return raw HTML as fallback
                return response.text, doc_type, is_academic
            
        else:
            return f"[Document type '{doc_type}' is not supported for extraction]", doc_type, is_academic
            
    except requests.exceptions.Timeout:
        print(f"Timeout error while accessing {url}")
        return f"[Error: Request timed out]", "unknown", False
    except requests.exceptions.ConnectionError:
        print(f"Connection error while accessing {url}")
        return f"[Error: Connection failed]", "unknown", False
    except requests.exceptions.TooManyRedirects:
        print(f"Too many redirects while accessing {url}")
        return f"[Error: Too many redirects]", "unknown", False
    except requests.exceptions.RequestException as e:
        print(f"Request error while accessing {url}: {e}")
        return f"[Error: {str(e)}]", "unknown", False
    except Exception as e:
        print(f"Unexpected error extracting text from {url}: {e}")
        return f"[Error: {str(e)}]", "unknown", False
def filter_relevant_content(content, query, threshold=0.1, max_paragraphs=15):  # Reduced threshold, increased max paragraphs
    if not content or len(content.strip()) < 100:
        return content

    # Clean up content before processing
    content = content.replace('\r\n', '\n')
    content = re.sub(r'\n{3,}', '\n\n', content)  # Normalize multiple newlines
    
    # Try different paragraph splitting strategies
    paragraphs = re.split(r'\n\s*\n', content)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # If we don't have enough paragraphs, try alternative splitting
    if len(paragraphs) < 5:
        paragraphs = re.split(r'(?:\r?\n){2,}', content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # If still not enough paragraphs, create artificial ones from sentences
        if len(paragraphs) < 5:
            try:
                sentences = re.split(r'(?<=[.!?])\s+', content)
                # Group sentences into paragraphs of ~5 sentences each
                new_paragraphs = []
                for i in range(0, len(sentences), 5):
                    paragraph = ' '.join(sentences[i:i+5])
                    if paragraph.strip():
                        new_paragraphs.append(paragraph)
                if len(new_paragraphs) >= 3:  # Only use if we get enough paragraphs
                    paragraphs = new_paragraphs
            except Exception as e:
                print(f"Error creating artificial paragraphs: {e}")
    
    # If we still don't have enough paragraphs, just return the content
    if len(paragraphs) < 3:
        print(f"Not enough paragraphs to filter content (found {len(paragraphs)})")
        return content
    
    try:
        # Score paragraphs for relevance
        scored_paragraphs = score_paragraph_relevance(paragraphs, query)
    except Exception as e:
        print(f"Error scoring paragraphs: {e}")
        return content
    
    # Extract keywords from query
    try:
        query_keywords = extract_keywords_from_query(query)
    except Exception as e:
        print(f"Error extracting keywords from query: {e}")
        query_keywords = []
    
    # Extract keywords from the top paragraphs
    try:
        # Get top paragraphs safely
        top_paragraphs = [p for p, score in scored_paragraphs[:min(5, len(scored_paragraphs))] if score > 0.05]
        top_content = "\n".join(top_paragraphs) if top_paragraphs else paragraphs[0]
        content_keywords = extract_keywords_from_text(top_content)
    except Exception as e:
        print(f"Error extracting keywords from content: {e}")
        content_keywords = []
    
    # Combine keywords
    try:
        combined_keywords = list(set(query_keywords + content_keywords))
        combined_keywords = [kw for kw in combined_keywords if kw and len(kw) > 2]
        
        if not combined_keywords and query:
            # Fallback to using words from the query if keyword extraction failed
            combined_keywords = [word.lower() for word in query.split() if len(word) > 3]
    except Exception as e:
        print(f"Error processing keywords: {e}")
        combined_keywords = []
    
    if combined_keywords:
        print(f"Keywords for filtering: {', '.join(combined_keywords[:10])}{' and more' if len(combined_keywords) > 10 else ''}")
    
    # Select relevant paragraphs
    try:
        relevant_paragraphs = []
        for p, score in scored_paragraphs:
            # Check for relevance score
            if score >= threshold:
                relevant_paragraphs.append(p)
                continue
                
            # Check for keyword matches if score is below threshold
            if combined_keywords:
                keyword_count = sum(1 for keyword in combined_keywords if keyword.lower() in p.lower())
                if keyword_count >= 1:  # Reduced keyword threshold from 2 to 1
                    relevant_paragraphs.append(p)
                    continue
            
            # Check for query terms (if we have fewer paragraphs than we need)
            if len(relevant_paragraphs) < max_paragraphs * 0.7 and query:
                query_term_count = sum(1 for term in query.lower().split() if len(term) > 3 and term.lower() in p.lower())
                if query_term_count >= 1:  # Reduced term count threshold
                    relevant_paragraphs.append(p)
    except Exception as e:
        print(f"Error selecting relevant paragraphs: {e}")
        # Return a safe subset of paragraphs
        relevant_paragraphs = [p for p, _ in scored_paragraphs[:max_paragraphs]] if scored_paragraphs else paragraphs[:max_paragraphs]
    
    # Limit to maximum paragraphs
    relevant_paragraphs = relevant_paragraphs[:max_paragraphs]
    
    # Check if we have enough paragraphs; if not, include some top-scoring ones
    if len(relevant_paragraphs) < max(3, max_paragraphs * 0.5):
        print(f"Found only {len(relevant_paragraphs)} relevant paragraphs, adding more")
        # Add top-scoring paragraphs we haven't included yet
        for p, score in scored_paragraphs:
            if p not in relevant_paragraphs:
                relevant_paragraphs.append(p)
                if len(relevant_paragraphs) >= max(5, max_paragraphs * 0.7):
                    break
    
    # Additional check for headers
    try:
        # Look for paragraphs that might be headers (short lines that end with colon 
        # or have special formatting like starting with #)
        header_indices = []
        for i, paragraph in enumerate(paragraphs):
            is_header = False
            # Check if it's short and followed by a colon
            if len(paragraph) < 100 and (paragraph.endswith(':') or paragraph.startswith('#')):
                is_header = True
            # Markdown-style headers
            elif paragraph.startswith('###') or paragraph.startswith('##') or paragraph.startswith('# '):
                is_header = True
            
            if is_header:
                header_indices.append(i)
        
        # Include headers if their content is included
        for i in header_indices:
            # Find the next header index or the end of the paragraphs
            next_header = next((h for h in header_indices if h > i), len(paragraphs))
            
            # Check if any paragraph between this header and the next is in our relevant paragraphs
            section_paragraphs = paragraphs[i+1:next_header]
            has_relevant_content = any(p in relevant_paragraphs for p in section_paragraphs)
            
            # If there's relevant content and we haven't included the header, add it
            if has_relevant_content and paragraphs[i] not in relevant_paragraphs:
                relevant_paragraphs.append(paragraphs[i])
    except Exception as e:
        print(f"Error processing headers: {e}")
    
    # Re-sort paragraphs to maintain original order
    try:
        paragraph_indices = {p: i for i, p in enumerate(paragraphs)}
        relevant_paragraphs.sort(key=lambda p: paragraph_indices.get(p, 0))
    except Exception as e:
        print(f"Error sorting paragraphs: {e}")
    
    # Check if filtering removed too much content
    result = "\n\n".join(relevant_paragraphs)
    if len(result) < min(len(content) * 0.05, 300):  # Reduced threshold for "too little content"
        print("Filtering removed too much content, using original")
        return content
        
    return result
def search_and_load(query):
    # Get URLs from Google search
    try:
        urls = query_googling(query)
        if not urls or len(urls) == 0:
            print("No URLs found from search query")
            return [Document(page_content="No search results found for the query", metadata={"source": "No URL", "document_type": "unknown", "is_academic": False})]
    except Exception as e:
        print(f"Error searching Google: {e}")
        return [Document(page_content=f"Error searching: {str(e)}", metadata={"source": "No URL", "document_type": "unknown", "is_academic": False})]
    
    # Suppress only the specific SSL verification warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    documents = []
    successful_loads = 0
    
    for url in urls:
        # Skip invalid URLs
        if not url.startswith(('http://', 'https://')):
            print(f"Skipping invalid URL: {url}")
            continue
            
        try:
            # Extract text based on document type (PDF, Word, HTML, etc.)
            content, doc_type, is_academic = extract_text_from_document_url(url, query)
            
            # Skip if we got an error message but keep going if we don't have any documents yet
            if content.startswith('[Error:') and len(documents) >= 1:
                print(f"Error from {url}: {content}")
                continue
            elif content.startswith('[Error:') and len(documents) == 0:
                # For first document, if it's an error but not "blocked access", try to keep it
                if "website may be blocking" not in content:
                    print(f"Keeping error document since we have no others: {content}")
                    doc = Document(
                        page_content=f"Error accessing content: {content}",
                        metadata={
                            "source": url,
                            "document_type": "unknown",
                            "is_academic": False,
                            "score_map": {},
                            "summary": "Could not access this document"
                        }
                    )
                    documents.append(doc)
                continue
                
            # Skip empty or very short content
            if not content or len(content.strip()) < 50:
                print(f"Skipping {url}: insufficient content (length: {len(content) if content else 0})")
                continue
            
            # Limit content size to prevent processing extremely large documents
            if len(content) > 500000:  # ~500KB of text
                print(f"Trimming oversized document from {url} ({len(content)} chars)")
                content = content[:500000] + "\n\n[Content truncated due to size]"
                
            # Get paragraph scores for highlighting
            try:
                paragraphs = re.split(r'\n\s*\n', content)
                paragraphs = [p.strip() for p in paragraphs if p.strip()]
                
                # Handle documents with no clear paragraph structure
                if len(paragraphs) < 2:
                    # Try alternative paragraph splitting
                    paragraphs = re.split(r'(?:\r?\n){2,}', content)
                    paragraphs = [p.strip() for p in paragraphs if p.strip()]
                    
                    # If still no paragraphs, create artificial ones
                    if len(paragraphs) < 2:
                        sentences = re.split(r'(?<=[.!?])\s+', content)
                        # Group sentences into paragraphs of ~5 sentences each
                        paragraphs = []
                        for i in range(0, len(sentences), 5):
                            paragraph = ' '.join(sentences[i:i+5])
                            if paragraph.strip():
                                paragraphs.append(paragraph)
                
                scored_paragraphs = score_paragraph_relevance(paragraphs, query)
                
                # Create a score map for later highlighting
                score_map = {p: score for p, score in scored_paragraphs}
            except Exception as e:
                print(f"Error scoring paragraphs for {url}: {e}")
                # Create a fallback score map
                score_map = {content: 0.5}
            
            # Filter content for relevance to the search query
            # For academic papers/PDFs, be less aggressive with filtering to preserve structure
            try:
                if doc_type in ['pdf', 'docx', 'doc'] and is_academic:
                    threshold = 0.05  # Lower threshold for academic content
                    max_paragraphs = 25  # Keep more content from academic sources
                else:
                    threshold = 0.1  # Reduced from 0.15
                    max_paragraphs = 15  # Increased from 10
                    
                relevant_content = filter_relevant_content(content, query, threshold, max_paragraphs)
                
                # If filtering removed too much, use original content
                if not relevant_content or len(relevant_content) < (len(content) * 0.05):  # Reduced from 0.1
                    print(f"Filtering removed too much content for {url}, using original")
                    relevant_content = content
            except Exception as e:
                print(f"Error filtering content for {url}: {e}")
                relevant_content = content
            
            # Generate summary with error handling
            try:
                summary = generate_summary(relevant_content, query)
            except Exception as e:
                print(f"Error generating summary for {url}: {e}")
                summary = "Summary generation failed."
            
            # Add document to collection
            documents.append(Document(
                page_content=relevant_content, 
                metadata={
                    "source": url,
                    "document_type": doc_type,
                    "is_academic": is_academic,
                    "score_map": score_map,
                    "summary": summary
                })
            )
            
            successful_loads += 1
            
            # Break early if we have enough successful documents
            if successful_loads >= 5:
                break
                
        except Exception as e:
            print(f"Error processing {url}: {e}")
    
    # Fallback to WebBaseLoader if no content was extracted
    if not documents:
        try:
            print("No documents extracted, trying fallback loader")
            
            # Only try fallback for the first few URLs to avoid excessive time
            fallback_urls = urls[:3] if len(urls) > 3 else urls
            
            fallback_loader = WebBaseLoader(
                web_paths=fallback_urls,
                verify_ssl=False,
                requests_kwargs={
                    'timeout': 30,
                    'headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                }
            )
            fallback_documents = fallback_loader.load()
            filtered_documents = []
            
            # Filter fallback documents for relevance
            for doc in fallback_documents:
                if doc.page_content and len(doc.page_content.strip()) > 50:
                    try:
                        # Get paragraph scores for highlighting
                        paragraphs = re.split(r'\n\s*\n', doc.page_content)
                        paragraphs = [p.strip() for p in paragraphs if p.strip()]
                        scored_paragraphs = score_paragraph_relevance(paragraphs, query)
                        
                        # Create a score map for later highlighting
                        score_map = {p: score for p, score in scored_paragraphs}
                        
                        relevant_content = filter_relevant_content(doc.page_content, query)
                        
                        # If filtering removed too much, use original content
                        if not relevant_content or len(relevant_content) < (len(doc.page_content) * 0.05):  # Reduced from 0.1
                            relevant_content = doc.page_content
                            
                        # Generate summary
                        try:
                            summary = generate_summary(relevant_content, query)
                        except Exception as e:
                            print(f"Error generating fallback summary: {e}")
                            summary = "Summary generation failed."
                        
                        filtered_documents.append(Document(
                            page_content=relevant_content, 
                            metadata={
                                **doc.metadata,
                                "document_type": "html",
                                "is_academic": False,
                                "score_map": score_map,
                                "summary": summary
                            })
                        )
                    except Exception as e:
                        print(f"Error processing fallback document: {e}")
            
            if filtered_documents:
                documents = filtered_documents
            else:
                # Create a placeholder document with useful information about the error
                example_url = urls[0] if urls else "No URL"
                documents = [Document(
                    page_content="Could not retrieve content from the websites. Many websites block automated access. Try a different search query or add more specific terms.",
                    metadata={
                        "source": example_url,
                        "document_type": "html",
                        "is_academic": False,
                        "score_map": {},
                        "summary": "Failed to retrieve content from search results."
                    }
                )]
        except Exception as fallback_error:
            print(f"Error loading fallback documents: {fallback_error}")
            documents = [Document(
                page_content=f"Error retrieving search results: {str(fallback_error)}. Try a different search query.",
                metadata={
                    "source": urls[0] if urls else "No URL",
                    "document_type": "html",
                    "is_academic": False,
                    "score_map": {},
                    "summary": "Error retrieving content."
                }
            )]
    
    # Calculate average relevance scores with error handling
    for doc in documents:
        try:
            if "score_map" in doc.metadata and doc.metadata["score_map"]:
                doc.metadata["avg_score"] = sum(doc.metadata["score_map"].values()) / max(len(doc.metadata["score_map"]), 1)
            else:
                doc.metadata["avg_score"] = 0
        except Exception as e:
            print(f"Error calculating average score: {e}")
            doc.metadata["avg_score"] = 0
    
    # Sort documents by relevance score, with a preference for academic sources
    try:
        documents.sort(key=lambda x: (x.metadata.get("avg_score", 0) * (1.2 if x.metadata.get("is_academic", False) else 1.0)), reverse=True)
    except Exception as e:
        print(f"Error sorting documents: {e}")
    
    # Filter out documents with relevance score below threshold, unless we have too few documents
    if len(documents) > 3:
        documents = [doc for doc in documents if doc.metadata.get("avg_score", 0) > 0.01]  # Reduced from 0.03
    
    # Don't filter out "unknown" document types so aggressively
    # if len(documents) > 2:
    #     documents = [doc for doc in documents if doc.metadata.get("document_type", "unknown").lower() != "unknown"]
    
    # Ensure we return at least one document
    if not documents:
        documents = [Document(
            page_content="No relevant content found. Try refining your search query with more specific terms.",
            metadata={
                "source": "No URL",
                "document_type": "html",
                "is_academic": False,
                "score_map": {},
                "summary": "No results found."
            }
        )]
        
    return documents
tool = Tool(
    name="google_search_and_load",
    description="Search Google and load webpage content from results",
    func=search_and_load
)


def print_formatted_results(results):
    print("\n" + "=" * 100)
    print(f"{'SEARCH RESULTS':^100}")
    print("=" * 100 + "\n")
    
    document_count = 0
    
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source', 'Unknown')
        avg_score = doc.metadata.get('avg_score', 0)
        score_map = doc.metadata.get('score_map', {})
        summary = doc.metadata.get('summary', '')
        doc_type = doc.metadata.get('document_type', 'html').upper()
        is_academic = doc.metadata.get('is_academic', False)
        
        # Don't skip documents with type "UNKNOWN" anymore
        # if doc_type == "UNKNOWN":
        #     continue
            
        document_count += 1
        
        # Document type icon
        type_icon = "📄"  # Default
        if doc_type == 'PDF':
            type_icon = "📑"
        elif doc_type in ('DOCX', 'DOC'):
            type_icon = "📝"
        elif doc_type == 'PRESENTATION':
            type_icon = "🎭"
        elif doc_type == 'SPREADSHEET':
            type_icon = "📊"
        elif doc_type == 'UNKNOWN':
            type_icon = "❓"
            
        # Academic badge
        academic_badge = "🎓 ACADEMIC SOURCE" if is_academic else ""
        
        print(f"\n{'▓' * 100}")
        print(f"{type_icon} DOCUMENT {i} | TYPE: {doc_type} | {academic_badge}")
        print(f"SOURCE: {source} | RELEVANCE: {avg_score:.2f}")
        
        # Extract and display bibliographic information if available for academic sources
        if is_academic:
            biblio_info = extract_bibliographic_info(doc.page_content, doc_type.lower(), is_academic)
            if biblio_info:
                print(f"{'▓' * 100}\n")
                print("📚 BIBLIOGRAPHIC INFORMATION:")
                if biblio_info.get("title"):
                    print(f"TITLE: {biblio_info['title']}")
                if biblio_info.get("authors"):
                    print(f"AUTHORS: {biblio_info['authors']}")
                if biblio_info.get("publication"):
                    print(f"PUBLICATION: {biblio_info['publication']}")
                if biblio_info.get("year"):
                    print(f"YEAR: {biblio_info['year']}")
                if biblio_info.get("doi"):
                    print(f"DOI: {biblio_info['doi']}")
                print()
            else:
                print(f"{'▓' * 100}\n")
        else:
            print(f"{'▓' * 100}\n")
        
        # Print summary in a highlighted box if available
        if summary and summary != doc.page_content:
            print("📋 KEY POINTS:")
            print("┌" + "─" * 98 + "┐")
            
            # Wrap summary text to fit in the box
            wrapped_summary = textwrap.fill(summary, width=96)
            for line in wrapped_summary.split('\n'):
                print(f"│ {line:<96} │")
                
            print("└" + "─" * 98 + "┘\n")
        
        # Format content based on document type
        if doc_type in ('PDF', 'DOCX', 'DOC') and is_academic:
            # Use special formatting for academic papers to preserve structure
            content = doc.page_content
            content = content.replace('\r\n', '\n').strip()
            
            # Format page markers in PDFs
            content = re.sub(r'--- Page (\d+) ---', r'📄 PAGE \1 ', content)
            
            # Preserve the structure more for academic content
            print(content)
        else:
            # Use regular formatting for web content
            print(format_content(doc.page_content, score_map))
            
        print("\n" + "▬" * 100 + "\n")
    
    # Print a message if no documents were displayed
    if document_count == 0:
        print("No valid documents found in the search results.")
    
    print("\n" + "▬" * 100 + "\n")

# Example usage.
def runs(query):
    results = tool.run(query)
    print_formatted_results(results)
    # print_formatted_results(results)
    return results

# # Test with a query
# if __name__ == "__main__":
#     runs("o3 mini openai model capabilities")