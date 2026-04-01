import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.lib.hybrid_search import HybridSearch
from src.lib.medicine_data import load_cached_docs
from src.config import config as cfg

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.lib.utils import tokenise_string

MINI_DOC_METADATA = {
    "006911": {
        "id": "006911","name": "Zavesca", "url_code": "zavesca",
        "therapeutic_area": ["Gaucher Disease"],
        "active_substance": ["miglustat"], "atc_code": "n06ba02",
    },
    "000419": {
        "id": "000419", "name": "Glivec", "url_code": "glivec",
        "therapeutic_area": ["Chronic Myeloid Leukaemia"],
        "active_substance": ["imatinib"], "atc_code": "l01xe01",
    },
    "000552": {
        "id": "000552", "name": "MabThera", "url_code": "mabthera",
        "therapeutic_area": ["Non-Hodgkin's Lymphoma"],
        "active_substance": ["rituximab"], "atc_code": "l01xc02",
    },
    "000278": {
        "id": "000278", "name": "Herceptin", "url_code": "herceptin",
        "therapeutic_area": ["Breast Cancer"],
        "active_substance": ["trastuzumab"], "atc_code": "l01xc03",
    },
    "000481": {
        "id": "000481", "name": "Humira", "url_code": "humira",
        "therapeutic_area": ["Rheumatoid Arthritis"],
        "active_substance": ["adalimumab"], "atc_code": "l04ab04",
    },
    "000771": {
        "id": "000771", "name": "Glucophage", "url_code": "glucophage",
        "therapeutic_area": ["Type 2 Diabetes Mellitus"],
        "active_substance": ["metformin hydrochloride"], "atc_code": "a10ba02",
    },
    "000523": {
        "id": "000523", "name": "Lipitor", "url_code": "lipitor",
        "therapeutic_area": ["Hypercholesterolaemia"],
        "active_substance": ["atorvastatin"], "atc_code": "c10aa05",
    },
    "000166": {
        "id": "000166", "name": "Losec", "url_code": "losec",
        "therapeutic_area": ["Gastroesophageal Reflux Disease"],
        "active_substance": ["omeprazole"], "atc_code": "a02bc01",
    },
    "000106": {
        "id": "000106", "name": "Norvasc", "url_code": "norvasc",
        "therapeutic_area": ["Hypertension"],
        "active_substance": ["amlodipine"], "atc_code": "c08ca01",
    },
    "000258": {
        "id": "000258", "name": "Taxol", "url_code": "taxol",
        "therapeutic_area": ["Ovarian Cancer"],
        "active_substance": ["paclitaxel"], "atc_code": "l01cd01",
    },
}

MINI_CHUNK_METADATA = [
    # Zavesca (miglustat) — Gaucher Disease
    {
        "doc_id": "006911", "section": "4.1 Therapeutic indications",
        "chunk_text": (
            "Zavesca (miglustat) is indicated for the treatment of adult patients with mild to "
            "moderate type 1 Gaucher disease for whom enzyme replacement therapy is unsuitable."
        ),
    },
    {
        "doc_id": "006911", "section": "4.2 Posology and method of administration",
        "chunk_text": (
            "The recommended dose of miglustat is 100 mg three times daily. Dose reduction to "
            "100 mg once or twice daily may be considered in patients who experience adverse "
            "effects such as tremor or diarrhoea."
        ),
    },
    {
        "doc_id": "006911", "section": "4.3 Contraindications",
        "chunk_text": (
            "Miglustat is contraindicated in patients with hypersensitivity to the active "
            "substance or to any of the excipients. It is also contraindicated during pregnancy "
            "and in women of childbearing potential not using contraception."
        ),
    },
    # Glivec (imatinib) — CML
    {
        "doc_id": "000419", "section": "4.1 Therapeutic indications",
        "chunk_text": (
            "Glivec (imatinib) is indicated for the treatment of adult patients with Philadelphia "
            "chromosome positive chronic myeloid leukaemia (CML) in chronic phase after failure "
            "of interferon-alpha therapy, or in accelerated phase or blast crisis."
        ),
    },
    {
        "doc_id": "000419", "section": "4.2 Posology and method of administration",
        "chunk_text": (
            "The recommended dose of imatinib in chronic phase CML is 400 mg per day. The "
            "recommended dose in accelerated phase and blast crisis is 600 mg per day. Doses "
            "may be increased to 600 mg or 800 mg in patients not responding adequately."
        ),
    },
    {
        "doc_id": "000419", "section": "4.4 Special warnings",
        "chunk_text": (
            "Hepatotoxicity has been observed with imatinib therapy. Liver function should be "
            "monitored at baseline and monthly thereafter, or as clinically indicated. Cases of "
            "severe hepatic failure including fatalities have been reported."
        ),
    },
    # MabThera (rituximab) — NHL
    {
        "doc_id": "000552", "section": "4.1 Therapeutic indications",
        "chunk_text": (
            "MabThera (rituximab) is indicated for the treatment of adult patients with "
            "follicular lymphoma stage III-IV who are chemoresistant or are in their second or "
            "subsequent relapse after chemotherapy."
        ),
    },
    {
        "doc_id": "000552", "section": "4.2 Posology and method of administration",
        "chunk_text": (
            "Rituximab is administered as an intravenous infusion. The recommended dose for "
            "non-Hodgkin's lymphoma is 375 mg/m2 body surface area given as an intravenous "
            "infusion once weekly for four weeks."
        ),
    },
    {
        "doc_id": "000552", "section": "4.8 Undesirable effects",
        "chunk_text": (
            "Infusion-related reactions are the most common adverse effects of rituximab, "
            "occurring predominantly during the first infusion. Symptoms include fever, chills, "
            "rigors, hypotension, urticaria, angioedema, nausea, and fatigue."
        ),
    },
    # Herceptin (trastuzumab) — Breast Cancer
    {
        "doc_id": "000278", "section": "4.1 Therapeutic indications",
        "chunk_text": (
            "Herceptin (trastuzumab) is indicated for the treatment of adult patients with "
            "HER2-positive metastatic breast cancer. It is used as monotherapy or in combination "
            "with paclitaxel or docetaxel for first-line treatment."
        ),
    },
    {
        "doc_id": "000278", "section": "4.2 Posology and method of administration",
        "chunk_text": (
            "Trastuzumab loading dose is 8 mg/kg body weight administered as a 90-minute "
            "intravenous infusion. Maintenance dose is 6 mg/kg every three weeks administered "
            "as a 30-minute infusion if the loading dose was well tolerated."
        ),
    },
    {
        "doc_id": "000278", "section": "4.4 Special warnings",
        "chunk_text": (
            "Cardiotoxicity including symptomatic cardiac failure has been observed with "
            "trastuzumab. Cardiac function should be assessed prior to and during treatment. "
            "Use with caution in patients with a history of hypertension or coronary artery disease."
        ),
    },
    # Humira (adalimumab) — RA
    {
        "doc_id": "000481", "section": "4.1 Therapeutic indications",
        "chunk_text": (
            "Humira (adalimumab) is indicated for treatment of moderate to severe, active "
            "rheumatoid arthritis in adult patients when the response to disease-modifying "
            "antirheumatic drugs including methotrexate has been inadequate."
        ),
    },
    {
        "doc_id": "000481", "section": "4.2 Posology and method of administration",
        "chunk_text": (
            "The recommended dose of adalimumab for rheumatoid arthritis is 40 mg administered "
            "every other week as a single dose via subcutaneous injection. Methotrexate should "
            "be continued during treatment with Humira."
        ),
    },
    {
        "doc_id": "000481", "section": "4.3 Contraindications",
        "chunk_text": (
            "Adalimumab is contraindicated in patients with active tuberculosis or other severe "
            "infections. It is also contraindicated in moderate to severe heart failure (NYHA "
            "class III/IV) and in patients with hypersensitivity to the active substance."
        ),
    },
    # Glucophage (metformin) — Type 2 Diabetes
    {
        "doc_id": "000771", "section": "4.1 Therapeutic indications",
        "chunk_text": (
            "Glucophage (metformin hydrochloride) is indicated for the treatment of type 2 "
            "diabetes mellitus, particularly in overweight patients, when diet and exercise "
            "alone do not provide adequate glycaemic control."
        ),
    },
    {
        "doc_id": "000771", "section": "4.2 Posology and method of administration",
        "chunk_text": (
            "The usual starting dose of metformin is 500 mg or 850 mg two or three times daily "
            "given during or after meals. Dose should be increased gradually to reduce "
            "gastrointestinal side-effects. Maximum recommended dose is 3000 mg daily."
        ),
    },
    {
        "doc_id": "000771", "section": "4.3 Contraindications",
        "chunk_text": (
            "Metformin is contraindicated in patients with renal failure or renal dysfunction, "
            "hepatic impairment, acute or chronic conditions associated with tissue hypoxia, "
            "dehydration, excess alcohol intake, and diabetic ketoacidosis."
        ),
    },
    # Lipitor (atorvastatin) — Hypercholesterolaemia
    {
        "doc_id": "000523", "section": "4.1 Therapeutic indications",
        "chunk_text": (
            "Lipitor (atorvastatin) is indicated as adjunct to diet for reduction of elevated "
            "total cholesterol, LDL-cholesterol, apolipoprotein B, and triglycerides in adults "
            "with primary hypercholesterolaemia or mixed dyslipidaemia."
        ),
    },
    {
        "doc_id": "000523", "section": "4.2 Posology and method of administration",
        "chunk_text": (
            "The usual starting dose of atorvastatin is 10 mg once daily. Dose adjustments "
            "should be made at intervals of 4 weeks or more. The maximum dose is 80 mg once "
            "daily. Atorvastatin can be administered at any time of day, with or without food."
        ),
    },
    {
        "doc_id": "000523", "section": "4.8 Undesirable effects",
        "chunk_text": (
            "Myopathy and rhabdomyolysis have been reported in patients receiving atorvastatin. "
            "Patients should be advised to report promptly unexplained muscle pain, tenderness, "
            "or weakness. Liver enzyme elevations have been reported rarely."
        ),
    },
    # Losec (omeprazole) — GERD
    {
        "doc_id": "000166", "section": "4.1 Therapeutic indications",
        "chunk_text": (
            "Losec (omeprazole) is indicated for the treatment of gastroesophageal reflux "
            "disease, including erosive reflux oesophagitis. It is also indicated for the "
            "prevention of relapse in patients with healed reflux oesophagitis."
        ),
    },
    {
        "doc_id": "000166", "section": "4.2 Posology and method of administration",
        "chunk_text": (
            "For treatment of reflux oesophagitis, the recommended dose of omeprazole is "
            "20 mg once daily for 4 weeks. Patients not fully healed should be treated for "
            "a further 4 weeks. Maintenance dose is 10 mg once daily."
        ),
    },
    {
        "doc_id": "000166", "section": "4.5 Interactions",
        "chunk_text": (
            "Omeprazole may increase the plasma concentrations of drugs metabolised by "
            "CYP2C19, such as warfarin, clopidogrel, and diazepam. Monitoring of INR is "
            "recommended when omeprazole is combined with warfarin or coumarin anticoagulants."
        ),
    },
    # Norvasc (amlodipine) — Hypertension
    {
        "doc_id": "000106", "section": "4.1 Therapeutic indications",
        "chunk_text": (
            "Norvasc (amlodipine) is indicated for the treatment of hypertension and for the "
            "symptomatic treatment of chronic stable angina and vasospastic angina. It may be "
            "used as monotherapy or in combination with other antihypertensive agents."
        ),
    },
    {
        "doc_id": "000106", "section": "4.2 Posology and method of administration",
        "chunk_text": (
            "For hypertension and angina, the usual initial dose of amlodipine is 5 mg once "
            "daily. The dose may be increased to a maximum of 10 mg once daily depending on "
            "individual patient response. Amlodipine may be taken with or without food."
        ),
    },
    {
        "doc_id": "000106", "section": "4.4 Special warnings",
        "chunk_text": (
            "Patients with severe aortic stenosis should be treated with caution. Amlodipine "
            "should be used with caution in patients with hepatic impairment. Dose reduction "
            "may be required in patients with severe hepatic impairment."
        ),
    },
    # Taxol (paclitaxel) — Ovarian Cancer
    {
        "doc_id": "000258", "section": "4.1 Therapeutic indications",
        "chunk_text": (
            "Taxol (paclitaxel) is indicated for the treatment of ovarian cancer in combination "
            "with cisplatin, and for treatment of node-positive breast cancer following standard "
            "doxorubicin and cyclophosphamide combination chemotherapy."
        ),
    },
    {
        "doc_id": "000258", "section": "4.2 Posology and method of administration",
        "chunk_text": (
            "For ovarian cancer, the recommended dose of paclitaxel is 175 mg/m2 administered "
            "intravenously over three hours, followed by cisplatin 75 mg/m2. Courses should "
            "be repeated every three weeks."
        ),
    },
    {
        "doc_id": "000258", "section": "4.8 Undesirable effects",
        "chunk_text": (
            "Myelosuppression, particularly neutropenia, is the dose-limiting toxicity of "
            "paclitaxel. Peripheral neuropathy, alopecia, arthralgia, myalgia, and "
            "hypersensitivity reactions are also commonly observed adverse effects."
        ),
    },
]


@pytest.fixture(scope="session")
def mini_engine():
    """
    Lightweight HybridSearch built from 10 medicines (~30 chunks).
    No cache files are read or written.
    """

    engine = HybridSearch.__new__(HybridSearch)

    # Attributes expected by ChunkedSemanticSearch / SemanticSearch parents
    engine.chunk_metadata = MINI_CHUNK_METADATA
    engine.doc_metadata = MINI_DOC_METADATA
    engine.docmap = {}

    # Lightweight model (~80 MB vs 1.3 GB for BGE)
    engine.model = SentenceTransformer("all-MiniLM-L6-v2")

    chunk_texts = [c["chunk_text"] for c in MINI_CHUNK_METADATA]
    engine.chunk_embeddings = engine.model.encode(chunk_texts, show_progress_bar=False)

    tokenized = [tokenise_string(t) for t in chunk_texts]
    engine.bm25 = BM25Okapi(tokenized)

    return engine


# ---------------------------------------------------------------------------
# Integration fixture — real data loaded from cache
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def app_setup():
    print("Loading documents...")
    try:
        documents = load_cached_docs()
    except Exception:
        print("No documents found, rebuilding from PDFs.")
        raise Exception("No documents found")

    if not documents:
        raise Exception("No documents found")
    
    app.state.search_engine = HybridSearch(documents=documents)
    
    yield app
    
    print("Shutting down...")


@pytest.fixture
def client(app_setup):
    app = app_setup
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Unit tests — use mini_engine directly, no HTTP overhead
# ---------------------------------------------------------------------------

def test_mini_rrf_search_returns_results(mini_engine):
    results = mini_engine.rrf_search("miglustat dose", limit=5)
    doc_ids = [doc["id"] for doc in results]
    assert len(results) > 0
    assert "006911" in doc_ids
    


def test_mini_rrf_search_result_fields(mini_engine):
    results = mini_engine.rrf_search("dose", limit=3)
    assert len(results) > 0
    first = results[0]
    for field in ("id", "name", "section", "text", "RRF", "BM25", "SEM"):
        assert field in first, f"Missing field: {field}"


def test_mini_rrf_search_respects_limit(mini_engine):
    results = mini_engine.rrf_search("treatment dose", limit=2)
    assert len(results) <= 2


def test_mini_rrf_filter_by_therapeutic_area(mini_engine):
    results = mini_engine.rrf_search(
        "dose", limit=10, therapeutic_area="Hypertension"
    )
    assert len(results) > 0
    for r in results:
        assert r["id"] == "000106"


def test_mini_rrf_filter_by_active_substance(mini_engine):
    results = mini_engine.rrf_search(
        "dose", limit=10, active_substance="imatinib"
    )
    assert len(results) > 0
    for r in results:
        assert r["id"] == "000419"


def test_mini_rrf_filter_by_atc_prefix(mini_engine):
    # "l01" matches imatinib (l01xe01), rituximab (l01xc02), trastuzumab (l01xc03), paclitaxel (l01cd01)
    results = mini_engine.rrf_search("treatment cancer", limit=10, atc_code="l01")
    assert len(results) > 0
    for r in results:
        assert MINI_DOC_METADATA[r["id"]]["atc_code"].startswith("l01")


def test_mini_rrf_filter_no_match_returns_empty(mini_engine):
    results = mini_engine.rrf_search(
        "dose", limit=5, therapeutic_area="Nonexistent Disease XYZ"
    )
    assert results == []


def test_mini_rrf_ranking_scores_positive(mini_engine):
    results = mini_engine.rrf_search("contraindication hypersensitivity", limit=5)
    for r in results:
        assert r["RRF"] > 0


# ---------------------------------------------------------------------------
# Integration / HTTP tests — require full app_setup
# ---------------------------------------------------------------------------

def test_app_health(client):
    response = client.get("/healthz/")
    assert response.status_code == 200
    assert "Healthy" in response.json()


def test_search_basic_query(client):
    response = client.post("/chats/", json={"query": "miglustat dose", "limit": 5})
    assert response.status_code == 200
    data = response.json()


def test_search_with_filters(client):
    response = client.post("/chats/", json={
        "query": "miglustat dose",
        "therapeutic_area": "Gaucher Disease",
        "limit": 10,
    })
    assert response.status_code == 200
    assert "Norvasc" not in response.json()
    
    


def test_search_empty_query(client):
    response = client.post("/chats/", json={"query": ""})
    assert response.status_code == 400
