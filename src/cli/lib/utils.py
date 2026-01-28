import json
import string
from pathlib import Path
from nltk.stem import PorterStemmer


stemmer = PorterStemmer()

cache_path = Path(__file__).parent.parent.parent / "cache"
embeddings_path = cache_path / "drug_embeddings.npy"
metadata_path = cache_path / "drug_metadata.json"
documents_path = cache_path / "drug_docs.json"

def load_stopwords():
    stopwords_path = "/Users/yasseryaya-oye/workspace/hybridgreen/dawa/data/stopwords.txt"
    try:
        with open(stopwords_path, "r") as f:
            data = f.read()
            words = data.splitlines()
            return words
    except Exception as e:
        print(f"Error loading stopwords: {str(e)}")


def load_cached_docs():
    try:
        with open(documents_path, "r") as f:
            data = json.load(f)
            print(f"Loaded {len(data)} documents data")
            return data
    except Exception as e:
        print(f"Error: {str(e)}")


def tokenise_string(input: str):
    
    stopwords = load_stopwords()
    punct = string.punctuation
    input = list(input.lower())
    output = []

    for i in range(len(input)):
        if input[i] not in punct:
            output.append(input[i])

    output = "".join(output).split(" ")

    output = [t for t in output if t]
    output = [t for t in output if t not in stopwords]
    output = [stemmer.stem(t) for t in output]

    return output


def normalise_score(score, min_scores, max_scores):
    if min_scores == max_scores:
        return 1.0
    return (score - min_scores) / (max_scores - min_scores)




