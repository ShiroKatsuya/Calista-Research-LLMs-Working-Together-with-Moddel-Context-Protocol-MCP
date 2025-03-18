import os
import sys
import requests
import urllib3
from urllib.parse import urlparse, unquote
import io
import fitz  # PyMuPDF
import tempfile

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_document(url):
    """Download a document from a URL"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml,application/pdf,*/*',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    
    print(f"Downloading from: {url}")
    response = requests.get(url, headers=headers, verify=False, timeout=30, stream=True)
    response.raise_for_status()
    
    # Determine file type
    content_type = response.headers.get('Content-Type', '').lower()
    filename = os.path.basename(urlparse(url).path)
    
    if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
        print(f"Detected PDF document: {filename}")
        document_type = 'pdf'
    elif 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type or url.lower().endswith('.docx'):
        print(f"Detected DOCX document: {filename}")
        document_type = 'docx'
    else:
        print(f"Unknown document type: {content_type}")
        document_type = 'unknown'
    
    return response.content, document_type, filename

def extract_text_from_pdf(pdf_content):
    """Extract text from PDF content"""
    try:
        # Open PDF from binary content
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        print(f"PDF document has {pdf_document.page_count} pages")
        
        # Extract text from each page
        text = []
        for page_num in range(min(5, pdf_document.page_count)):  # Only first 5 pages for test
            page = pdf_document[page_num]
            page_text = page.get_text()
            text.append(f"\n--- Page {page_num + 1} ---\n")
            text.append(page_text[:500] + "..." if len(page_text) > 500 else page_text)  # Limit text for preview
            
        pdf_document.close()
        return "\n".join(text)
    except Exception as e:
        return f"Error extracting PDF text: {e}"

def test_pdf_extraction():
    """Test downloading and extracting text from a PDF"""
    # Use an open-access academic PDF
    pdf_url = "https://arxiv.org/pdf/1601.06759.pdf"  # This is a random academic paper
    
    try:
        pdf_content, doc_type, filename = download_document(pdf_url)
        
        if doc_type == 'pdf':
            text = extract_text_from_pdf(pdf_content)
            print("\n=========== EXTRACTED TEXT PREVIEW ===========\n")
            print(text[:1000])  # Show first 1000 chars as preview
            print("\n==============================================\n")
            print(f"Total extracted text length: {len(text)} characters")
        else:
            print(f"Not a PDF document: {doc_type}")
            
    except Exception as e:
        print(f"Error during PDF testing: {e}")

if __name__ == "__main__":
    print("Testing PDF extraction functionality...")
    test_pdf_extraction() 