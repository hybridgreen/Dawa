from typing import TypedDict, Literal, NotRequired, List
from pathlib import Path
import pymupdf
import pymupdf4llm
import json
from datetime import datetime, timedelta
import requests
import requests
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
import time


cache_path = Path(__file__).parent.parent.parent / "cache"
metadata_path = cache_path / "medicine_metadata.json"


def extract_markdown(pdf_path: str, ema_number: str):
    md_text = pymupdf4llm.to_markdown(pdf_path)

    md_path = Path(__file__).parent.parent.parent.parent / (
        f"./data/markdown/{ema_number}.md"
    )
    Path(md_path).parent.mkdir(parents=True, exist_ok=True)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    return md_text

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

def verify_pdf(pdf_path: str) -> bool:
    
    try:
        file_size = Path(pdf_path).stat().st_size
        if file_size < 1024: 
            return False
        
        with open(pdf_path, 'rb') as f:
            header = f.read(4)
            if not header.startswith(b'%PDF'):
                return False
        
        doc = pymupdf.open(pdf_path)
        
        if len(doc) == 0:
            doc.close()
            return False
        
        try:
            first_page = doc[0]
            _ = first_page.get_text("text")
        except Exception:
            doc.close()
            return False
        
        doc.close()
        return True
        
    except Exception as e:
        print(f"PDF verification failed: {e}")
        return False

def download_med_data(med_data_url: str) -> str:
        
    print("Downloading Medicine Data")
    file_path = Path(__file__).parent.parent.parent.parent / (
        "./data/medicine_data_en.json"
    )
    headers = {"user-agent": "dawa/0.0.1"}

    res = requests.get(med_data_url, stream=True, timeout=30, headers=headers)

    if res.ok:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        body = res.json()
        with open(file_path, "w") as f:
            json.dump(body, f, indent= 2)
        
        med_data_path =  str(file_path)
        
    if med_data_path and Path(med_data_path).exists():
        print("Medicine Data downloaded.")
        return med_data_path
    else:
        raise Exception(f"Medical Data file download failed. HTTP-{res.status_code}")

def download_pdfs(data_path: str, n_rows: int = 0):
    
    doc_metadata = load_existing_metadata(metadata_path)
    
    with open(data_path, 'r') as f:
        data = json.load(f)['data']
    
    medicines_to_process = data[:n_rows] if n_rows > 0 else data
    
    dl_count = 0
    
    for raw in medicines_to_process:
        try:
            medicine_name = raw['name_of_medicine']
            ema_number = raw['ema_product_number'].split("/")[-1]
            url_code = raw['medicine_url'].split('/')[-1]
            
            if skip_download(ema_number, raw, doc_metadata):
                print(f"Skipping {medicine_name} (up-to-date)")
                continue
            
            print(f"Downloading {medicine_name} ({ema_number})...")
            pdf_url = f"https://www.ema.europa.eu/en/documents/product-information/{url_code.lower()}-epar-product-information_en.pdf"
            
            pdf_path = fetch_pdf(pdf_url, f"pdf/{ema_number}-en", "pdf")
            
            if not pdf_path or not verify_pdf(pdf_path):
                raise Exception("Download or verification failed")
            
            metadata = construct_metadata(raw)
            metadata['updated_at'] = datetime.now().date().isoformat()
            doc_metadata[ema_number] = metadata
            
            dl_count += 1
            print(f"✓ Success")
            
        except Exception as e:
            print(f" ✗ Error processing {raw.get('name_of_medicine', 'unknown')}: {e}")
            
            try:
                metadata = construct_metadata(raw)
                metadata['updated_at'] = None
                doc_metadata[ema_number] = metadata
            except:
                pass 
            
            continue
        
    save_metadata(doc_metadata, metadata_path)
    
    print(f"\n✓ Downloaded {dl_count}/{len(medicines_to_process)} documents")
    
    return dl_count

def fetch_pdf(
    url: str, filename: str = "file", extension: str = "pdf", max_retries: int = 3
):
    file_path = Path(__file__).parent.parent.parent.parent / (
        f"./data/{filename}.{extension}"
    )

    headers = {"user-agent": "dawa/0.0.1"}

    for attempt in range(1, max_retries+1):
        res = requests.get(url, stream=True, timeout=30, headers=headers)
        

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


def skip_download(ema_number: str, raw: dict, doc_metadata: dict) -> bool:
    
    if ema_number not in doc_metadata:
        return False
    
    metadata = doc_metadata[ema_number]
    
    updated_at = metadata.get('updated_at')
    if not updated_at:
        return False 
    
    last_update_str = raw.get('last_updated_date', '').strip()
    if not last_update_str:
        return True
    
    try:
        local_date = datetime.fromisoformat(updated_at).date()
        source_date = datetime.strptime(last_update_str, "%d/%m/%Y").date()
        
        offset = timedelta(days=1) 
        return source_date < local_date + offset
        
    except ValueError:
        return False 


def save_metadata(doc_metadata: dict, metadata_path: str):
    
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(doc_metadata, f, indent=2, default=str)
    print(f"Saved metadata to {metadata_path}")


def load_existing_metadata(metadata_path: str) -> dict:
    if Path(metadata_path).exists():
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing metadata: {e}")
    return {}


class MedicineMetadata(TypedDict):
    # Required fields
    category: Literal["Human", "Veterinary"]
    name_of_medicine: str
    ema_product_number: str
    status: Literal["Authorised", "Withdrawn", "Suspended", "Refused", "Not authorised", "Opinion", "Application withdrawn", "Opinion under re-examination"]
    active_substance: str
    therapeutic_area_mesh: str
    atc_code_human: str
    revision_number: str
    last_update: str
    created_at: str
    updated_at: str
    medicine_url: str

    
    # Optional/can be empty
    opinion_status: NotRequired[str]
    latest_procedure_affecting_product_information: NotRequired[str]
    international_non_proprietary_name_common_name: NotRequired[str]
    species_veterinary: NotRequired[str]
    atcvet_code_veterinary: NotRequired[str]
    pharmacotherapeutic_group_human: NotRequired[str]
    pharmacotherapeutic_group_veterinary: NotRequired[str]
    therapeutic_indication: NotRequired[str]
    marketing_authorisation_developer_applicant_holder: NotRequired[str]
    
    # Boolean flags
    patient_safety: bool
    accelerated_assessment: bool
    additional_monitoring: bool
    advanced_therapy: bool
    biosimilar: bool
    conditional_approval: bool
    exceptional_circumstances: bool
    generic_or_hybrid: bool
    orphan_medicine: bool
    prime_priority_medicine: bool
    

def construct_metadata(raw: dict) -> MedicineMetadata:
    
    def to_bool(value) -> bool:
        return str(value).strip() == 'Yes'

    ema_number: str = raw['ema_product_number'].split("/")[-1]

    return {
        'id': ema_number,
        'category': raw['category'],
        'name': raw['name_of_medicine'],
        'status': raw['medicine_status'],
        'therapeutic_area': [
            area.strip() 
            for area in raw['therapeutic_area_mesh'].lower().split(';')
            if area.strip()
        ],
        'active_substance': [
            substance.strip() 
            for substance in raw['active_substance'].lower().split(';')
            if substance.strip()
        ],
        'atc_code': raw['atc_code_human'].lower(),
        
        # Convert all boolean flags
        'patient_safety': to_bool(raw['patient_safety']),
        'accelerated_assessment': to_bool(raw['accelerated_assessment']),
        'additional_monitoring': to_bool(raw['additional_monitoring']),
        'advanced_therapy': to_bool(raw['advanced_therapy']),
        'biosimilar': to_bool(raw['biosimilar']),
        'conditional_approval': to_bool(raw['conditional_approval']),
        'exceptional_circumstances': to_bool(raw['exceptional_circumstances']),
        'generic_or_hybrid': to_bool(raw['generic_or_hybrid']),
        'orphan_medicine': to_bool(raw['orphan_medicine']),
        'prime_priority_medicine': to_bool(raw['prime_priority_medicine']),
        
        'url': raw['medicine_url'],
        'last_update': raw['last_updated_date'],
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }

