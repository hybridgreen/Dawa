import pandas as pd
import requests
from pathlib import Path
import time
import pymupdf4llm
import json
from langchain_text_splitters import MarkdownTextSplitter
import re




def fetch_url(url:str, filename:str = "file", extension: str = "pdf"):
    
    file_path = Path(__file__).parent.parent.parent.parent / (f"./data/{filename}.{extension}")

    headers = {'user-agent': 'dawa/0.0.1'}
    
    res = requests.get( url,
                       stream= True,
                       timeout = 30,
                       headers = headers)
    
    if res.ok:    
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb+') as f:
            for chunk in res.iter_content(chunk_size = 8192):
                f.write(chunk)
        return file_path
    else:
        raise Exception(f"HTTP Error - {res.status_code}")


def process_pdfs(data_path: str):
    
    splitter = MarkdownTextSplitter(chunk_size=100, chunk_overlap=20)
    
    medicine = pd.read_excel(data_path, skiprows=8, nrows= 10 )

    data = medicine[["Name of medicine", "EMA product number", "Therapeutic area (MeSH)","Active substance","Revision number" , "Medicine URL", ]]
    
    md_docs = {}
    metadatas = {}
    
    for idx, row in data.iterrows():
        
        medicine_name: str = row["Name of medicine"]
        ema_number: str = row["EMA product number"].split("/")[-1]
        
        print(f"Processing {ema_number}...")
        
        try :
            pdf_path = fetch_url(
                f"https://www.ema.europa.eu/en/documents/product-information/{medicine_name.lower()}-epar-product-information_en.pdf",
                f"pdf/{ema_number}-en",
                "pdf")
            time.sleep(0.5)
            
            if pdf_path:
                
                md_text = pymupdf4llm.to_markdown(pdf_path)
                md_docs[ema_number]= md_text
                metadata = {
                "ema_number": ema_number,
                "medicine_name": row["Name of medicine"],
                "therapeutic_area": row["Therapeutic area (MeSH)"],
                "active_substance": row["Active substance"],
                "url": row["Medicine URL"]
                }
                metadatas[ema_number] = metadata
                
                      
        except Exception as e:
            print("Error", str(e))
            continue
    
    save_file(md_docs, "documents", "json")
    save_file(metadatas, "metadata", "json")
    
def extract_markdown(pdf_path: str, ema_number: str):
    
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    md_path = Path(__file__).parent.parent.parent.parent / (f"./data/markdown/{ema_number}.md")
    Path(md_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_text)
    
    return md_text

def save_file(data, filename:str, extension: str):
    
    filepath = Path(__file__).parent.parent.parent.parent / (f"./data/{filename}.{extension}")
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return filepath
    
        
        
        