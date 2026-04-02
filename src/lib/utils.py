import re
from nltk.stem import PorterStemmer
from pathlib import Path

stemmer = PorterStemmer()

cache_path = Path(__file__).parent.parent / "cache"
embeddings_path = cache_path / "drug_embeddings.npy"
metadata_path = cache_path / "drug_metadata.json"
stopwords_path =cache_path / "stopwords.txt"

def load_stopwords():
    
    if stopwords_path.exists():
        try:
            with open(stopwords_path, "r") as f:
                data = f.read()
                words = data.splitlines()
                return words
        except Exception as e:
            print(f"Error loading stopwords: {str(e)}")
    else:
        raise Exception("Unable to load stopwords")


def tokenise_string(text: str) -> list[str]:
    text = text.lower()

    text = re.sub(r"\*+", " ", text)
    tokens = re.findall(r"\b\w+\b", text)

    stopwords = load_stopwords()

    tokens = [t for t in tokens if t not in stopwords]
    tokens = [t for t in tokens if len(t) > 1]
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens
