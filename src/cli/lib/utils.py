import json
import string
from pathlib import Path
from nltk.stem import PorterStemmer
import requests
from pathlib import Path
import pymupdf4llm


stemmer = PorterStemmer()

cache_path = Path(__file__).parent.parent.parent / "cache"
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

import pandas as pd


def fetch_url(url:str, filename:str = "file", extension: str = "pdf"):
    
    file_path = Path(__file__).parent.parent.parent.parent / (f"./data/{filename}.{extension}")

    headers = {'user-agent': 'dawa/0.0.1'}
    
    res = requests.get( url,
                       stream= True,
                       timeout = 30,
                       headers = headers)
    
    if res.ok:    
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            for chunk in res.iter_content(chunk_size = 8192):
                f.write(chunk)
        return file_path
    else:
        raise Exception(f"HTTP Error - {res.status_code}  ")


def pdf_to_md(pdf_path: str):
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        print(f"File not found: {pdf_path}")
        return
    markdown = pymupdf4llm.to_markdown(pdf_file)
    return markdown


from pathlib import Path

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
                ema_number = pdf_file.name.split('-')[0]
                results.append({
                    'id': str(ema_number),
                    'content': markdown.lower()
                })
                print(f"  ✓ Extracted {len(markdown)} characters")
            else:
                print(f"  ✗ Failed to extract")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    print(f"\nSuccessfully processed {len(results)}/{len(pdf_files)} files")
    with open(documents_path, 'w') as f:
        json.dump(results, f, indent= 2)
    return results

def load_cached_docs():
    try:
        with open(documents_path, "r") as f:
            data = json.load(f)
            print(f"Loaded {len(data)} documents data")
            return data
    except Exception as e:
        print(f"Error: {str(e)}")