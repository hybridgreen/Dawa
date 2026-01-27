from typing import TypedDict, Literal, NotRequired, List
from .utils import fetch_pdf
from pathlib import Path
import pymupdf
import pymupdf4llm
import json
from datetime import datetime
import requests

cache_path = Path(__file__).parent.parent.parent / "cache"
metadata_path = cache_path / "drug_metadata.json"


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

def download_pdfs(data_path: str ,n_rows:int):
    
    dl_count = 0

    try:
        with open(data_path, "r") as f:
            raw_data = json.load(f)
            if raw_data:
                print(f"Loaded medical data")
            else:
                raise Exception("Failed to load medical data")
            
    except Exception as e:
        print(f"Error: {str(e)}")
    
    meta = raw_data['meta']
    data :  List[MedicineMetadata] = raw_data['data']
            
    if Path(metadata_path).exists():
        try:
            with open(metadata_path, "r") as f:
                doc_metadata: dict = json.load(f)
        except Exception:
            pass
    else:
        doc_metadata = {}
    
    if n_rows == 0:
        n_rows = meta['total_records']

    for data in data[:n_rows]:
        
        medicine_name = data['name_of_medicine']
        url_code = data['medicine_url'].split('/')[-1]
        ema_number: str = data['ema_product_number'].split("/")[-1]
        try:
            last_update =  datetime.strptime(data['last_updated_date'], "%d/%m/%Y")
        except Exception as e:
            last_update = None
            
        
        if ema_number in doc_metadata:
            metadata = doc_metadata[ema_number]
        else:
            metadata = None
        
        updated_at = datetime.fromisoformat(metadata['updated_at']) if metadata and metadata.get('updated_at') else None
        
        if not metadata or not updated_at or not last_update or last_update > updated_at:
            
            try:
                print(f"Downloading data for {medicine_name}, number: {ema_number}...")
                pdf_path = fetch_pdf (
                    f"https://www.ema.europa.eu/en/documents/product-information/{url_code.lower()}-epar-product-information_en.pdf",
                    f"pdf/{ema_number}-en",
                    "pdf",
                )
                
                if pdf_path :
                    if not verify_pdf(pdf_path):
                        raise Exception("Invalid pdf file - Use download command again") 
                    data['created_at'] = str(datetime.now())
                    data['updated_at'] = str(datetime.now())
                    
                    doc_metadata[str(ema_number)] = data
                    
                    print("Success")
                    dl_count += 1  
                    
            except Exception as e:
                print(f"Error - Downloading {medicine_name}, number: {ema_number}: {str(e)}")
                data['created_at'] = str(datetime.now())
                data['updated_at'] = None
                doc_metadata[ema_number] = data
                continue

    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(doc_metadata, f, indent=2, default=str)
    
    print(f"Successfully downloaded {dl_count} docs")
    

class MedicineMetadata(TypedDict):
    # Required fields
    category: Literal["Human", "Veterinary"]
    name_of_medicine: str
    ema_product_number: str
    medicine_status: Literal["Authorised", "Withdrawn", "Suspended", "Refused", "Not authorised", "Opinion", "Application withdrawn", "Opinion under re-examination"]
    active_substance: str
    therapeutic_area_mesh: str
    atc_code_human: str
    revision_number: str
    last_updated_date: str
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
    patient_safety: Literal["Yes", "No"]
    accelerated_assessment: Literal["Yes", "No"]
    additional_monitoring: Literal["Yes", "No"]
    advanced_therapy: Literal["Yes", "No"]
    biosimilar: Literal["Yes", "No"]
    conditional_approval: Literal["Yes", "No"]
    exceptional_circumstances: Literal["Yes", "No"]
    generic_or_hybrid: Literal["Yes", "No"]
    orphan_medicine: Literal["Yes", "No"]
    prime_priority_medicine: Literal["Yes", "No"]
    
    # Dates (optional - can be empty string)
    european_commission_decision_date: NotRequired[str]
    start_of_rolling_review_date: NotRequired[str]
    start_of_evaluation_date: NotRequired[str]
    opinion_adopted_date: NotRequired[str]
    withdrawal_of_application_date: NotRequired[str]
    marketing_authorisation_date: NotRequired[str]
    refusal_of_marketing_authorisation_date: NotRequired[str]
    withdrawal_expiry_revocation_lapse_of_marketing_authorisation_date: NotRequired[str]
    suspension_of_marketing_authorisation_date: NotRequired[str]
    first_published_date: NotRequired[str]
    
    