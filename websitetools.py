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
os.environ["GOOGLE_CSE_ID"] = "5650b4d81ac344bd8"
google_api1 = "AIzaSyAsaqSFC6hQrSpUhZTkt1GhCyNcw-StC7A"
google_api2 = "AIzaSyBEl28l8XlkhYWxCkGamWLAWW2Jfc3SJR0"
GOOGLE_API_KEY = google_api1
os.environ["USER_AGENT"] = "jfE7aT4JW0dDcqCLyoiVfQ==kblDif2SgvOKHDhQ"
def extract_text_from_document_url(url, query):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml,application/pdf,*/*',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        response = requests.get(url, headers=headers, verify=False, timeout=30, stream=True)
        response.raise_for_status()
        doc_type = detect_document_type(url, response)
        is_academic = is_academic_source(url)
        
        metadata = {
            "source": url,
            "document_type": doc_type,
            "is_academic": is_academic
        }
        if doc_type == 'pdf':
            content = extract_text_from_pdf(response.content)
            return content, doc_type, is_academic
            
        elif doc_type in ['docx', 'doc']:
            content = extract_text_from_docx(response.content)
            return content, doc_type, is_academic
            
        elif doc_type == 'txt':
            return response.text, doc_type, is_academic
            
        elif doc_type == 'html':
            content = extract_main_content(response.text, url)
            return content, doc_type, is_academic
            
        else:
            return f"[Document type '{doc_type}' is not supported for extraction]", doc_type, is_academic
            
    except Exception as e:
        print(f"Error extracting text from {url}: {e}")
        return f"[Error: {str(e)}]", "unknown", False
def filter_relevant_content(content, query, threshold=0.15, max_paragraphs=10):


    paragraphs = re.split(r'\n\s*\n', content)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    

    if len(paragraphs) < 5:
        return content
    

    scored_paragraphs = score_paragraph_relevance(paragraphs, query)
    

    query_keywords = extract_keywords_from_query(query)
    

    top_content = "\n".join([p for p, score in scored_paragraphs[:5] if score > 0.05])
    content_keywords = extract_keywords_from_text(top_content)

    combined_keywords = list(set(query_keywords + content_keywords))
    

    combined_keywords = [kw for kw in combined_keywords if len(kw) > 2]
    
    print(f"Automatically extracted keywords for filtering: {', '.join(combined_keywords)}")

    relevant_paragraphs = []
    for p, score in scored_paragraphs:

        keyword_count = sum(1 for keyword in combined_keywords if keyword.lower() in p.lower())

        if score >= threshold or keyword_count >= 2:
            relevant_paragraphs.append(p)
    

    relevant_paragraphs = relevant_paragraphs[:max_paragraphs]
    

    header_indices = []
    for i, paragraph in enumerate(paragraphs):
        if paragraph.startswith('###'):
            header_indices.append(i)
    

    for i in header_indices:

        next_header = next((h for h in header_indices if h > i), len(paragraphs))
        has_relevant_content = any(p in relevant_paragraphs for p in paragraphs[i+1:next_header])
        
        if has_relevant_content and paragraphs[i] not in relevant_paragraphs:
            relevant_paragraphs.append(paragraphs[i])
    
    # Re-sort paragraphs to maintain original order
    paragraph_indices = {p: i for i, p in enumerate(paragraphs)}
    relevant_paragraphs.sort(key=lambda p: paragraph_indices.get(p, 0))
    
    return "\n\n".join(relevant_paragraphs)
def search_and_load(query):
    # Get URLs from Google search
    urls = query_googling(query)
    
    # Suppress only the specific SSL verification warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    documents = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    for url in urls:
        try:
            # Extract text based on document type (PDF, Word, HTML, etc.)
            content, doc_type, is_academic = extract_text_from_document_url(url, query)
            
            # Skip empty or very short content
            if not content or len(content) < 50:
                print(f"Skipping {url}: insufficient content")
                continue
                
            # Get paragraph scores for highlighting
            paragraphs = re.split(r'\n\s*\n', content)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            scored_paragraphs = score_paragraph_relevance(paragraphs, query)
            
            # Create a score map for later highlighting
            score_map = {p: score for p, score in scored_paragraphs}
            
            # Filter content for relevance to the search query
            # For academic papers/PDFs, be less aggressive with filtering to preserve structure
            if doc_type in ['pdf', 'docx', 'doc'] and is_academic:
                threshold = 0.10  # Lower threshold for academic content
                max_paragraphs = 20  # Keep more content from academic sources
            else:
                threshold = 0.15
                max_paragraphs = 10
                
            relevant_content = filter_relevant_content(content, query, threshold, max_paragraphs)
            
            if relevant_content:
                # Generate summary focusing on content related to query
                summary = generate_summary(relevant_content, query)
                
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
            
        except Exception as e:
            print(f"Error processing {url}: {e}")
    
    # Fallback to WebBaseLoader if no content was extracted
    if not documents:
        try:
            fallback_loader = WebBaseLoader(
                web_paths=urls,
                verify_ssl=False
            )
            fallback_documents = fallback_loader.load()
            filtered_documents = []
            
            # Filter fallback documents for relevance
            for doc in fallback_documents:
                if doc.page_content.strip():
                    # Get paragraph scores for highlighting
                    paragraphs = re.split(r'\n\s*\n', doc.page_content)
                    paragraphs = [p.strip() for p in paragraphs if p.strip()]
                    scored_paragraphs = score_paragraph_relevance(paragraphs, query)
                    
                    # Create a score map for later highlighting
                    score_map = {p: score for p, score in scored_paragraphs}
                    
                    relevant_content = filter_relevant_content(doc.page_content, query)
                    if relevant_content:
                        # Generate summary focusing on content related to query
                        summary = generate_summary(relevant_content, query)
                        
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
            
            if filtered_documents:
                documents = filtered_documents
            else:
                documents = [Document(page_content="No relevant content found", metadata={"source": urls[0] if urls else "No URL", "document_type": "unknown", "is_academic": False})]
        except Exception as fallback_error:
            print(f"Error loading fallback documents: {fallback_error}")
            documents = [Document(page_content=f"Error: {str(fallback_error)}", metadata={"source": urls[0] if urls else "No URL", "document_type": "unknown", "is_academic": False})]
    
    # Sort documents by their average paragraph relevance score
    for doc in documents:
        if "score_map" in doc.metadata and doc.metadata["score_map"]:
            doc.metadata["avg_score"] = sum(doc.metadata["score_map"].values()) / len(doc.metadata["score_map"])
        else:
            doc.metadata["avg_score"] = 0
    
    # Sort documents by relevance score, with a preference for academic sources
    documents.sort(key=lambda x: (x.metadata.get("avg_score", 0) * (1.2 if x.metadata.get("is_academic", False) else 1.0)), reverse=True)
    
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
    
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source', 'Unknown')
        avg_score = doc.metadata.get('avg_score', 0)
        score_map = doc.metadata.get('score_map', {})
        summary = doc.metadata.get('summary', '')
        doc_type = doc.metadata.get('document_type', 'html').upper()
        is_academic = doc.metadata.get('is_academic', False)
        
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

# Example usage.
def runs(query):
    results = tool.run(query)
    print_formatted_results(results)

# runs("apa dampak jika tidur hanya 4 jam sehari")
