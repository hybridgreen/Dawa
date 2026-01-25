import json
import string
from pathlib import Path
from nltk.stem import PorterStemmer
import requests
import pymupdf4llm
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
import time
import pandas as pd
from datetime import datetime

stemmer = PorterStemmer()

cache_path = Path(__file__).parent.parent.parent / "cache"
embeddings_path = cache_path / "drug_embeddings.npy"
metadata_path = cache_path / "drug_metadata.json"
documents_path = cache_path / "drug_docs.json"


def load_file_data(filename: str, extension: str):
    file_path = Path(__file__).parent.parent.joinpath(f"./data/{filename}.{extension}")

    match extension:
        case "json":
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    print(f"Loaded {filename} data")
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


def fetch_url(
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
            return file_path
        
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


def pdf_to_md(pdf_path: str):
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        print(f"File not found: {pdf_path}")
        return
    markdown = pymupdf4llm.to_markdown(pdf_file)
    return markdown


def process_all_pdfs(folder_path: str):
    pdf_dir = Path(folder_path)

    if not pdf_dir.exists():
        print(f"Folder not found: {folder_path}")
        return []

    pdf_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))

    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return []

    print(f"Found {len(pdf_files)} PDF files")

    results = []

    for pdf_file in pdf_files:
        try:
            print(f"\nProcessing: {pdf_file.name}")

            markdown = pdf_to_md(str(pdf_file)).lower()

            if markdown:
                ema_number = pdf_file.name.split("-")[0]
                results.append({"id": str(ema_number), "content": markdown.lower()})
                print(f"✓ Extracted {len(markdown)} characters")
            else:
                print("✗ Failed to extract")
                

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    print(f"\nSuccessfully processed {len(results)}/{len(pdf_files)} files")
    
    with open(documents_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def load_cached_docs():
    try:
        with open(documents_path, "r") as f:
            data = json.load(f)
            print(f"Loaded {len(data)} documents data")
            return data
    except Exception as e:
        print(f"Error: {str(e)}")


def refresh_documents():
    
    med_data_path = "/Users/yasseryaya-oye/workspace/hybridgreen/dawa/data/medicine_data_en.xlsx"
    
    
    with open(metadata_path, "r") as f:
        doc_metadata: dict = json.load(f)

    if med_data_path:
        data = pd.read_excel(med_data_path, skiprows=8, nrows = len(doc_metadata.items()))[
            [
                "Name of medicine",
                "EMA product number",
                "Therapeutic area (MeSH)",
                "Active substance",
                "Revision number",
                "Medicine URL",
                "Last updated date"
            ]
        ]
    
    for idx, row in data.iterrows():
        
        medicine_name: str = row["Name of medicine"]
        url_code = str(row["Medicine URL"]).split('/')[-1]
        ema_number: str = row["EMA product number"].split("/")[-1]
        
        metadata = doc_metadata[ema_number]
        
        last_update =  datetime.strptime(row["Last updated date"], "%d/%m/%Y")
        
        if metadata['updated_at'] == None or last_update > datetime.now():
        
            try:
                print(f"Updating data for {medicine_name}, number: {ema_number}...")
                pdf_path = fetch_url(
                    f"https://www.ema.europa.eu/en/documents/product-information/{url_code.lower()}-epar-product-information_en.pdf",
                    f"pdf/{ema_number}-en",
                    "pdf",
                )
                
                if pdf_path:
                    print("Success")
                    doc_metadata[ema_number]['updated_at'] = str(datetime.now())
                    doc_metadata[ema_number]['last_update'] = str(last_update)
                    

            except Exception as e:
                print(f"Failed to update {medicine_name}, number : {ema_number}")
                continue

    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(doc_metadata, f, indent=2)
