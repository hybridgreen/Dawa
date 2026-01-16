import typer
from pathlib import Path
import pandas as pd
from llama_index.core import Document
from lib.data_fetch import fetch_url, process_pdfs

app = typer.Typer()

@app.command()
def download():
    file_path = fetch_url("https://www.ema.europa.eu/en/documents/report/medicines-output-medicines-report_en.xlsx", "medicine_data_en", "xlsx" )
    print("Doanloaded Med data, saved at: ", file_path )

@app.command()
def process(filepath: str):
    process_pdfs(filepath)
    pass
    
    
if __name__ == "__main__":
    app()
