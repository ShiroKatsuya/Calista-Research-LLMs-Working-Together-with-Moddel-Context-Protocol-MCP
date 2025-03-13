from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import re
def extract_keywords_from_query(query, num_keywords=8):

    # Tokenize the query
    try:
        # Try to determine if query is in Indonesian
        indonesian_markers = ['mengapa', 'kenapa', 'bagaimana', 'apa', 'siapa', 'kapan', 'dimana', 
                             'di', 'dan', 'atau', 'dengan', 'pada', 'untuk', 'dari', 'dalam',
                             'yang', 'adalah', 'ini', 'itu', 'oleh', 'jika', 'maka']
        
        is_indonesian = any(marker in query.lower().split() for marker in indonesian_markers)
        
        # Get stopwords for filtering - use English by default
        stop_words = set(stopwords.words('english'))
        
        # Add some common words that aren't useful as keywords
        additional_stops = {'why', 'how', 'what', 'when', 'where', 'who', 'which', 
                           'is', 'are', 'am', 'was', 'were', 'be', 'been', 'being',
                           'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could',
                           'will', 'would', 'shall', 'should', 'may', 'might', 'must',
                           'and', 'or', 'but', 'if', 'then', 'else', 'when', 'up', 'down',
                           'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
                           'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each',
                           'few', 'more', 'most', 'other', 'some', 'such'}
        
        # Add Indonesian stopwords if the query seems to be in Indonesian
        if is_indonesian:
            indonesian_stopwords = indonesian_markers + [
                'bahwa', 'sebagai', 'ia', 'mereka', 'saya', 'kamu', 'kita', 'kami',
                'dia', 'nya', 'itu', 'ini', 'ke', 'ya', 'juga', 'dapat', 'akan', 'masih',
                'telah', 'harus', 'secara', 'tetapi', 'tidak', 'belum', 'lebih', 'sangat',
                'bisa', 'tersebut', 'tentang', 'demikian', 'ketika', 'namun', 'menjadi',
                'seperti', 'sehingga', 'hingga'
            ]
            additional_stops.update(indonesian_stopwords)
            
        stop_words.update(additional_stops)
        
        # Tokenize and filter out stopwords
        words = word_tokenize(query.lower())
        keywords = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 2]
        
        # Count word frequencies
        word_counts = Counter(keywords)
        
        # Get the most common words
        common_words = [word for word, count in word_counts.most_common(num_keywords)]
        
        # If we don't have enough keywords, try to extract n-grams (2-word phrases)
        if len(common_words) < 3:
            # Get bigrams
            tokens = [token.lower() for token in words if token.isalnum()]
            bigrams = [" ".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
            bigram_counts = Counter(bigrams)
            top_bigrams = [bg for bg, count in bigram_counts.most_common(3) if len(bg) > 5]
            
            # Add individual words from bigrams
            for bigram in top_bigrams:
                words_in_bigram = bigram.split()
                for word in words_in_bigram:
                    if word not in common_words and word not in stop_words and len(word) > 2:
                        common_words.append(word)
        
        return common_words
    except Exception as e:
        print(f"Keyword extraction error: {e}")
        # Fallback to simple extraction if NLP processing fails
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        return list(set(words))[:num_keywords]