import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
import os
search = GoogleSearchAPIWrapper(k=1)


os.environ["GOOGLE_CSE_ID"] = "5650b4d81ac344bd8"
os.environ["GOOGLE_API_KEY"] = "AIzaSyBEl28l8XlkhYWxCkGamWLAWW2Jfc3SJR0"
os.environ["USER_AGENT"] = "jfE7aT4JW0dDcqCLyoiVfQ==kblDif2SgvOKHDhQ"

def search_and_load(query):
    results = search.results(query, 1)
    urls = []
    for result in results:
        urls.append(result['link'])
    return urls

urls = search_and_load("What is the capital of France?")

print(urls)


tool = Tool(
    name="google_search_and_load",
    description="Search Google and load webpage content from results",
    func=search_and_load
)

loader = WebBaseLoader(
    web_paths=(
     "https://en.wikipedia.org/wiki/Paris"
    ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

docs = text_splitter.split_documents(documents)

print(docs[1])
