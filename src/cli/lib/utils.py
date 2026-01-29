import re
from nltk.stem import PorterStemmer
from pathlib import Path

stemmer = PorterStemmer()

cache_path = Path(__file__).parent.parent.parent / "cache"
embeddings_path = cache_path / "drug_embeddings.npy"
metadata_path = cache_path / "drug_metadata.json"

def load_stopwords():
    stopwords_path = "/Users/yasseryaya-oye/workspace/hybridgreen/dawa/data/stopwords.txt"
    try:
        with open(stopwords_path, "r") as f:
            data = f.read()
            words = data.splitlines()
            return words
    except Exception as e:
        print(f"Error loading stopwords: {str(e)}")



stemmer = PorterStemmer()

def tokenise_string(text: str) -> list[str]:
    """
    Tokenize and normalize text for BM25.
    Handles markdown, medical symbols, and punctuation.
    """
    text = text.lower()
    
    text = re.sub(r'\*+', ' ', text)
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'\s+', ' ', text) 

    
    medical_symbols = {
        'µg': 'microgram',
        'µl': 'microliter',
        'mg': 'milligram',
        'ml': 'milliliter',
        '°c': 'degrees celsius',
        '≥': 'greater than equal',
        '≤': 'less than equal',
        '±': 'plus minus',
        '%': 'percent'
    }
    
    for symbol, word in medical_symbols.items():
        text = text.replace(symbol, f' {word} ')
    
    tokens = re.findall(r'\b\w+\b', text)
    stopwords = load_stopwords()
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [t for t in tokens if len(t) > 1]
    tokens = [stemmer.stem(t) for t in tokens]
    
    return tokens

def normalise_score(score, min_scores, max_scores):
    if min_scores == max_scores:
        return 1.0
    return (score - min_scores) / (max_scores - min_scores)




