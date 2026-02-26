# DawaAI
Healthcare professionals in West Africa face unique challenges when accessing pharmaceutical information:

- Connectivity Issues: Unreliable internet in remote clinics limits real-time drug lookups
- Information Overload: Thousands of complex pharmaceutical documents are difficult to navigate
- Time Pressure: Busy clinics need instant access to dosing, contraindications, and safety information

Dawa AI solves these problems by providing a fast, intelligent search system that works offline and delivers precise pharmaceutical information in seconds, not minutes.

## ✨ Key Features
### 🔍 Hybrid Search Engine

- Semantic Understanding: BGE-Large embeddings capture contextual meaning ("pediatric dose" matches "children dosing")
- Keyword Precision: BM25 ensures exact terms aren't missed ("100mg" finds exact dosages)
- Reciprocal Rank Fusion: Optimally combines semantic and lexical results for maximum relevance

### 🌐 Offline-First Architecture

- Pre-computed Embeddings: All 37k document sections embedded locally (no internet required for search)
- Instant Startup: Ready to serve queries in under 5 seconds

### 📊 Smart Filtering & Search

Filter by Therapeutic Area, Active Substance, direct ATC Classification: Pharmaceutical classification codes, Section-Specific

### 🔗 Direct Source Access

- Citation Links: Every result links directly to source EMA documents
- Section References: Precise section numbers (4.2 Posology, 4.3 Contraindications)
- Relevance Scoring: Transparent BM25, semantic, and hybrid scores

### 🚀 Production-Ready

- FastAPI Backend: High-performance async API
- Docker Containerized: Consistent deployment across environments
- Cloud Native with CLI Support: Google Cloud Run with auto-scaling

## Phase 2: Enhanced Intelligence

[] Multi-language Support: French language search for Francophone West Africa
[] Agentic Search: A simple text query runs filtered search
[] Voice Search: Audio queries for hands-free operation in clinical settings

## Tech Stack

Backend : FastAPI, Uvicorn
Search & ML : Sentence Transformers: BAAI/BGE-Large-EN-v1.5, rank-bm25: Okapi BM25 implementation for keyword search, NumPy
Data Processing: PyMuPDF: PDF text extraction and document parsing
Infrastructure: Docker, uv(Astral), GCP: Google Cloud Run, Artifact Registry


## Design Decisions

#### 1. Hybrid Search Architecture & RRF vs Weighted Average
Decision: Combine semantic (BGE) + lexical (BM25) search with Reciprocal Rank Fusion

Rationale:
Semantic-only misses exact medical terms (drug names, dosages)
Keyword-only misses contextual relationships (synonyms, related concepts)
RRF fusion empirically outperforms weighted averaging by 15-23%

Trade-offs:

- ✅ Pro: Best retrieval accuracy for medical queries
- ✅ Pro: Handles both precise and conceptual searches
- ✅ Pro: No hyperparameter tuning required
- ✅ Pro: Robust across different query types
- ✅ Pro: Well-established in information retrieval research
- ❌ Con: Slightly more complex to implement than weighted sum
- ❌ Con: Less intuitive than score-based combination
- ❌ Con: 2x computational overhead vs. single method
- ❌ Con: More complex caching and optimization

Alternative Considered: Linear combination (α × semantic + (1-α) × BM25)
Rejected: Requires tuning α, sensitive to score normalization

#### 2. Pre-computed Document Embeddings
Decision: Embed all documents offline, only embed queries in real-time
Rationale:

BGE-Large inference: 50ms per query vs. 20 minutes for full corpus
Offline requirement: No internet needed after initial deployment
Cost optimization: Avoid repeated embedding of static documents

Trade-offs:

- ✅ Pro: Instant search without model loading
- ✅ Pro: Works completely offline
- ✅ Pro: Predictable query latency
- ❌ Con: Large deployment artifacts (6GB total)
- ❌ Con: Document updates require full re-deployment

Alternative Considered: Real-time document embedding
Rejected: 1000x slower, requires GPU, defeats offline-first goal

#### 3. Docker + Cloud Run Deployment
Decision: Containerize with Docker, deploy on Google Cloud Run
Rationale:
Serverless scaling: 0-10 instances
Global reach & User Testing: Multi-region deployment for low latency
Cost efficiency: Pay only for actual usage (~$5-15/month)
Developer experience: Simple deployments with gcloud run deploy

Trade-offs:
- ✅ Pro: Zero infrastructure management
- ✅ Pro: Automatic scaling and load balancing
- ✅ Pro: Built-in HTTPS, monitoring, and logging
- ❌ Con: 10s second cold starts (mitigated with min instances)
- ❌ Con: Vendor lock-in (mitigated by standard containers)

#### 4. BGE-Large vs. Smaller Models
Decision: Use BAAI/BGE-Large-EN-v1.5 despite 1.3GB size
Rationale:

- Medical Domain: BGE-Large trained on scientific/medical text
- Embedding Quality: 1024 dimensions capture nuanced relationships
- Benchmark Results: 5-7% better retrieval accuracy vs. all-MiniLM-L6-v2
- Deployment Reality: 1.3GB acceptable for cloud deployment

Trade-offs:
- ✅ Pro: Best-in-class retrieval quality for medical queries
- ✅ Pro: Handles complex pharmaceutical terminology
- ❌ Con: Large Docker images (6GB vs. 300MB with MiniLM)
- ❌ Con: Higher memory requirements (8GB vs. 2GB)

Alternative Considered: all-MiniLM-L6-v2 (80MB)

