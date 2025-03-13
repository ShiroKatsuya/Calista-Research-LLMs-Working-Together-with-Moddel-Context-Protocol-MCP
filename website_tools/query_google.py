
from langchain_google_community import GoogleSearchAPIWrapper
search = GoogleSearchAPIWrapper(k=5)
# results = tool.run("apa dampak jika tidur hanya 4 jam sehari")
def query_googling(query):
    # Modified to search for academic content 
    academic_query = query
    # Check if we need to add academic terms
    if not any(term in query.lower() for term in ["pdf", "research", "journal", "thesis", "dissertation", "paper", "academia", "article"]):
        academic_query = f"{query} (pdf OR research paper OR journal OR dissertation OR thesis)"
    
    results = search.results(academic_query, 5)
    urls = []
    for result in results:
        urls.append(result['link'])
    print(urls)
    return urls