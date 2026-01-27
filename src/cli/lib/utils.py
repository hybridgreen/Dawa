import json
import string
from pathlib import Path
from nltk.stem import PorterStemmer
import requests
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
import time

stemmer = PorterStemmer()

cache_path = Path(__file__).parent.parent.parent / "cache"
embeddings_path = cache_path / "drug_embeddings.npy"
metadata_path = cache_path / "drug_metadata.json"
documents_path = cache_path / "drug_docs.json"


def load_file_data(file_path: str | Path, extension: str):

    match extension:
        case "json":
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    print(f"Loaded medical data")
                    return data
            except Exception as e:
                print(f"Error: {str(e)}")
        case "txt":
            try:
                with open(file_path, "r") as f:
                    data = f.read()
                    words = data.splitlines()
                    return words
            except Exception as e:
                print(f"Error: {str(e)}")

        case "pdf":
            try:
                with open(file_path, "rb") as f:
                    data = f.read()
                    return data
            except Exception as e:
                print(f"Error: {str(e)}")

        case _:
            raise Exception("Invalid file extension")


def tokenise_string(input: str):
    stopwords = load_file_data("stopwords", "txt")
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


def fetch_pdf(
    url: str, filename: str = "file", extension: str = "pdf", max_retries: int = 5
):
    file_path = Path(__file__).parent.parent.parent.parent / (
        f"./data/{filename}.{extension}"
    )

    headers = {"user-agent": "dawa/0.0.1"}

    session = requests.Session()

    retries = Retry(
        total=max_retries,
        backoff_factor=2,
        status_forcelist=[502, 503, 504],
        allowed_methods={"GET"},
    )

    session.mount("https://", HTTPAdapter(max_retries=retries))

    for attempt in range(max_retries):
        res = session.get(url, stream=True, timeout=30, headers=headers)

        if res.ok:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=8192):
                    f.write(chunk)
            return str(file_path)
        
        elif res.status_code == 429:
            try:
                print(f"Rate limited")
                retry_after = res.headers.get("Retry-After")
                wait_time = int(retry_after)
            except Exception:
                wait_time = 60 * attempt

            print(f"Waiting {wait_time}s...")
            time.sleep(wait_time)
        else:
            raise Exception(f"HTTP Error - {res.status_code}  ")





def load_cached_docs():
    try:
        with open(documents_path, "r") as f:
            data = json.load(f)
            print(f"Loaded {len(data)} documents data")
            return data
    except Exception as e:
        print(f"Error: {str(e)}")


