from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def score_paragraph_relevance(paragraphs, query):

    if not paragraphs:
        return []
    

    corpus = [query] + paragraphs
    

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    

    query_vector = tfidf_matrix[0:1]
    paragraph_vectors = tfidf_matrix[1:]
    similarity_scores = cosine_similarity(query_vector, paragraph_vectors)[0]
    
    # Pair paragraphs with their scores and sort by score
    scored_paragraphs = list(zip(paragraphs, similarity_scores))
    scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
    
    return scored_paragraphs