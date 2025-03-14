
from langchain_google_community import GoogleSearchAPIWrapper
search = GoogleSearchAPIWrapper(k=10)
# results = tool.run("apa dampak jika tidur hanya 4 jam sehari")
def query_googling(query):
    # Modified to search for academic content 
    normal = query
    # Check if we need to add academic terms
    # if not any(term in query.lower() for term in ["pdf", "research", "journal", "thesis", "dissertation", "paper", "academia", "article"]):
    #     normal = f"{query} (pdf OR research paper OR journal OR dissertation OR thesis)"
    
    results = search.results(normal, 10)
    urls = []
    for result in results:
        urls.append(result['link'])
    print(urls)
    return urls