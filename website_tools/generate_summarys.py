from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import re
from sklearn.feature_extraction.text import TfidfVectorizer
def generate_summary(text, query, num_sentences=5):

    # Skip summarization if text is too short
    if len(text.split()) < 100:
        return text
        
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    

    if len(sentences) <= num_sentences:
        return text
    
    # Clean sentences
    clean_sentences = [re.sub(r'[^\w\s]', '', s.lower()) for s in sentences]
    
    # Create TF-IDF vectorizer and calculate sentence scores
    vectorizer = TfidfVectorizer(stop_words='english')
    
    try:
        # Add query to the corpus to calculate relevance
        corpus = clean_sentences + [query.lower()]
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Calculate similarity between query and each sentence
        query_vector = tfidf_matrix[-1]
        sentence_vectors = tfidf_matrix[:-1]
        similarities = cosine_similarity(query_vector, sentence_vectors)[0]
        
        # Get indices of top sentences based on similarity score
        top_indices = heapq.nlargest(num_sentences, 
                                     range(len(similarities)), 
                                     similarities.__getitem__)
        
        # Sort indices to maintain original order
        top_indices.sort()
        
        # Create summary
        summary = [sentences[i] for i in top_indices]
        
        return " ".join(summary)
    except:
        # Fallback to simple extraction if vectorization fails
        return " ".join(sentences[:num_sentences])