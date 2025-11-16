# RAG FAQ Utility

A lightweight Retrieval-Augmented Generation (RAG) system designed to automatically answer FAQ-style questions based on internal company documentation.  
The system includes:

- Document chunking  
- Embedding generation  
- FAISS-based vector search  
- LLM-based answer generation  
- Evaluation agent  
- Fully reproducible indexing/query workflow  

---

# Repository Structure
```
RAG_FAQ_utility/
│
├── data/
│ └── faq_document.txt # Source FAQ document (≥1000 words)
│
├── src/
│ ├── build_index.py # Index builder: chunk → embed → create FAISS index
│ ├── query.py # Query engine: retrieve → generate answer → produce JSON
│ ├── evaluator_agent.py # Evaluation pipeline for RAG output
│ └── init.py
│
├── storage/ (generated after running `python src/build_index.py`)
│ ├── chunks.json # Chunk metadata
│ ├── vectors.npy # Embedding matrix
│ ├── faiss.index # Persisted FAISS index 
│ ├── faiss_meta.json # Embedding/index metadata
│ └── ...
│
├── outputs/ (generated after running `python src/query.py`)
│ ├── sample_queries.json # Saved query results
│ ├── metrics.json # Query metrics (latency, tokens, etc.)
│ └── queries_evaluations.json # Evaluations from evaluator agent (generated after running `python src/evaluator_agent.py`)
│
├── reports/
│ └── report_en.md # Architecture & technical report
│
├── tests/
│ └── test_core.py # Basic tests for indexing + querying
│
├── .env.example # Template environment variables
├── requirements.txt # Python dependencies
└── README.md # This file
```

---

# Setup Instructions

## Python Version
- System tested with Python **3.9.7**

## Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Configure environment variables
Create a `.env` file in the project root folder, following the `.env.example` file

# Usage

Below is the full workflow for building the index, querying it, and evaluating the output.

## Build the Index
```bash
python src/build_index.py
```

If you want to customize the input, e.g.:
```bash
python src/build_index.py  --input data/faq_document.txt  --store_dir storage  --words_chunk_size 120 --overlap_size 20
```

## Run a Query
```bash
python src/query.py --question "How do I invite a new user?"
```

If you want specify custom `k` most relevant chunks to be used to generate the response:
```bash
python src/query.py --question "What is RBAC?" --k 5
```

The response will follow this output example (printed as JSON):
```json
{
  "user_question": "...",
  "system_answer": "...",
  "chunks_related": [
    { "id": ..., "topic": "...", "text": "...",... }
  ]
}
```

The system will also save:

- `outputs/sample_queries.json`
- `outputs/metrics.json`

## Evaluate Answers (Optional)
Reads all entries in `outputs/sample_queries.json` and evaluates them:

```bash
python src/evaluator_agent.py
```
Creates:
- `outputs/queries_evaluations.json`

Each entry contains:
```json
{
  "user_question": "...",
  "system_answer": "...",
  "evaluation": {
    "score": int,
    "reason": "..."
  }
}
```

# Tests
Run
```bash
pytest -q
```
This executes the tests defined in `tests/test_core.py`

# Technical Decisions

For the Technical decisions, please go to the file `reports\report_en.md` to have a comprehensive explanation of the overall choices and definitions.