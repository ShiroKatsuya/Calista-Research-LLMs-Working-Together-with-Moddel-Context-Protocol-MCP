import numpy as np
from nltk.corpus import stopwords
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
def extract_keywords_from_text(text, num_keywords=15):
    """
    Extract important keywords from text using TF-IDF.
    
    Args:
        text (str): The text to extract keywords from
        num_keywords (int): Maximum number of keywords to extract
        
    Returns:
        list: List of extracted keywords
    """
    try:
        # Check if text appears to be in Indonesian
        indonesian_markers = ['dan', 'yang', 'di', 'pada', 'dengan', 'untuk', 'dari', 'dalam',
                             'adalah', 'ini', 'itu', 'oleh', 'jika', 'maka',
                             'tidak', 'bisa', 'akan', 'telah', 'sudah', 'harus']
        
        # Count occurrences of Indonesian markers
        indonesian_count = 0
        for marker in indonesian_markers:
            indonesian_count += text.lower().count(' ' + marker + ' ')
        
        # Use English stopwords by default
        stop_words = set(stopwords.words('english'))
        
        # Add Indonesian stopwords if enough markers are found
        if indonesian_count > 3:
            indonesian_stopwords = indonesian_markers + [
                'bahwa', 'sebagai', 'ia', 'mereka', 'saya', 'kamu', 'kita', 'kami',
                'dia', 'nya', 'itu', 'ini', 'ke', 'ya', 'juga', 'dapat', 'akan', 'masih',
                'telah', 'harus', 'secara', 'tetapi', 'tidak', 'belum', 'lebih', 'sangat',
                'bisa', 'tersebut', 'tentang', 'demikian', 'ketika', 'namun', 'menjadi',
                'seperti', 'sehingga', 'hingga'
            ]
            stop_words.update(indonesian_stopwords)
        
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_df=0.9,
            min_df=1,
            max_features=500,
            stop_words=stop_words
        )
        
        # Create a dummy document if text is too short
        if len(text.split()) < 20:
            # Just use word frequency for short texts
            words = word_tokenize(text.lower())
            keywords = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 2]
            word_counts = Counter(keywords)
            return [word for word, count in word_counts.most_common(num_keywords)]
        
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        # Make sure there are enough sentences
        if len(sentences) < 2:
            # Split by newlines or punctuation if not enough sentences
            sentences = re.split(r'[.!?;:\n]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Fit the vectorizer
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Get feature names (terms)
            feature_names = vectorizer.get_feature_names_out()
            
            # Sum up TF-IDF scores for each term across all sentences
            tfidf_scores = np.sum(tfidf_matrix.toarray(), axis=0)
            
            # Create a dictionary of terms and their scores
            term_scores = {term: score for term, score in zip(feature_names, tfidf_scores)}
            
            # Sort terms by score and take the top ones
            sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Extract only the terms (without scores)
            top_keywords = [term for term, score in sorted_terms[:num_keywords] if len(term) > 2]
            
            return top_keywords
        except ValueError:
            # Fallback if vectorizer fails
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 2]
            word_counts = Counter(words)
            return [word for word, count in word_counts.most_common(num_keywords)]
    except Exception as e:
        print(f"Text keyword extraction error: {e}")
        # Ultimate fallback
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return list(set(words))[:num_keywords]