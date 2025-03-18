import os
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.documents import Document
import bs4
from langchain_community.document_loaders import WebBaseLoader
import re
import textwrap
import urllib3

os.environ["GOOGLE_CSE_ID"] = "5650b4d81ac344bd8"
os.environ["GOOGLE_API_KEY"] = "AIzaSyBEl28l8XlkhYWxCkGamWLAWW2Jfc3SJR0"
os.environ["USER_AGENT"] = "jfE7aT4JW0dDcqCLyoiVfQ==kblDif2SgvOKHDhQ"
search = GoogleSearchAPIWrapper(k=5)

def query_google(query):
    results = search.results(query, 5)
    urls = []
    for result in results:
        urls.append(result['link'])
    return urls

def search_and_load(query):
    # Get URLs from Google search
    urls = query_google(query)
    
    # Suppress only the specific SSL verification warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # First attempt: load webpage content with a filtered parser, targeting specific CSS classes.
    loader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
        verify_ssl=False  # Handle SSL verification issues.
    )
    
    try:
        documents = loader.load()
        documents = [doc for doc in documents if doc.page_content.strip()]
    except Exception as e:
        print(f"Error loading documents with filter: {e}")
        documents = []
    
    # If no relevant content was extracted, fallback to loading the full webpage content.
    # This fallback approach is inspired by best practices detailed in the
    # [milvus full text search guide](https://milvus.io/docs/id/full_text_search_with_langchain.md).
    if not documents:
        try:
            print("Fallback: loading full webpage content without filtering.")
            fallback_loader = WebBaseLoader(
                web_paths=urls,
                verify_ssl=False
            )
            fallback_documents = fallback_loader.load()
            fallback_documents = [doc for doc in fallback_documents if doc.page_content.strip()]
            if fallback_documents:
                documents = fallback_documents
            else:
                documents = [Document(page_content="No relevant content found", metadata={"source": urls[0] if urls else "No URL"})]
        except Exception as fallback_error:
            print(f"Error loading fallback documents: {fallback_error}")
            documents = [Document(page_content=f"Error: {str(fallback_error)}", metadata={"source": urls[0] if urls else "No URL"})]
    
    return documents

def format_content(content):
    """
    Formats the given content to be more presentable and readable.
    
    Args:
        content (str): The raw text content to format.
        
    Returns:
        str: The formatted text suitable for display.
    """
    # Remove excessive whitespace and normalize line endings
    content = content.replace('\r\n', '\n').strip()
    
    # Add horizontal line for separation
    separator = "-" * 80 + "\n"
    
    # Improved header formatting using markdown syntax
    # Detect headers that are in title case or all caps
    content = re.sub(
        r'(?m)^(?P<header>[A-Z][A-Z\s]{2,})$', 
        r'\n### \g<header> ###\n', 
        content
    )
    
    # Enhanced bullet point formatting
    content = re.sub(r'(?m)^[\s•-]+\s+', '  • ', content)
    
    # Split into paragraphs
    paragraphs = re.split(r'\n\s*\n', content)
    formatted_paragraphs = []
    
    for para in paragraphs:
        if not para.strip():
            continue
        
        # Handle code blocks by preserving them
        if para.startswith('```'):
            formatted_paragraphs.append(para)
            continue
        
        # Handle blockquotes by preserving them
        if para.startswith('>'):
            formatted_paragraphs.append(para)
            continue
        
        # Collapse any whitespace within the paragraph
        single_line = re.sub(r'\s+', ' ', para.strip())
        
        # If it's a header (starts with ###), don't wrap it
        if single_line.startswith('###'):
            formatted_paragraphs.append(single_line)
        else:
            # Wrap the paragraph to fit within 80 characters
            wrapped = textwrap.fill(
                single_line, 
                width=80, 
                initial_indent='', 
                subsequent_indent='    ' if not single_line.startswith('  • ') else '      '
            )
            formatted_paragraphs.append(wrapped)
    
    # Join paragraphs with double newlines and add separators
    formatted_text = separator + "\n\n".join(formatted_paragraphs) + "\n" + separator
    
    return formatted_text

# Create tool that combines search and web loading.
tool = Tool(
    name="google_search_and_load",
    description="Search Google and load webpage content from results",
    func=search_and_load
)

# Example usage.
results = tool.run("kenapa indonesia sangat korup 2025?")
print("\nFormatted Document Content:\n")

# If results is a list of Document objects, format and print each one's content
for i, doc in enumerate(results, 1):
    print(f"\nDocument {i} - Source: {doc.metadata.get('source', 'Unknown')}\n")
    print(format_content(doc.page_content))
    print("\n")
