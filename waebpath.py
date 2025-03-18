import bs4
import urllib3
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Suppress SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def load_webpage_content(url):
    try:
        # First attempt with specific CSS classes
        loader = WebBaseLoader(
            web_paths=[url],
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header", "article", "main", "content")
                )
            ),
            verify_ssl=False
        )
        documents = loader.load()
        documents = [doc for doc in documents if doc.page_content.strip()]
        
        # Fallback to full page content if no content found
        if not documents:
            print(f"Fallback: loading full webpage content for {url}")
            fallback_loader = WebBaseLoader(
                web_paths=[url],
                verify_ssl=False
            )
            documents = fallback_loader.load()
            documents = [doc for doc in documents if doc.page_content.strip()]
            
        if not documents:
            documents = [Document(page_content="No content found", metadata={"source": url})]
            
    except Exception as e:
        print(f"Error loading webpage {url}: {e}")
        documents = [Document(page_content=f"Error: {str(e)}", metadata={"source": url})]
        
    return documents

# Example usage
url = "https://en.wikipedia.org/wiki/Paris"
documents = load_webpage_content(url)

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

if docs:
    print(docs[0])  # Print first chunk
else:
    print("No documents found")