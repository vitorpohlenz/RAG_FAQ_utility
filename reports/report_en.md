# RAG FAQ Utility – System Report

## Architecture Overview

The **RAG_FAQ_utility** project implements a lightweight yet production-oriented Retrieval-Augmented Generation (RAG) pipeline designed for HR SaaS environments with documentation-heavy workflows. The goal is to automatically answer repetitive FAQ-style questions using internal documents, avoiding manual search and reducing support load.  

The system uses a modular approach that cleanly separates **indexing**, **retrieval**, **LLM-based generation**, and **evaluation**.

---

## Core Components

### Document Chunking & Preprocessing
- The system parses a large plain-text FAQ file.
- Uses a **sliding window chunking algorithm**, splitting the text into overlapping word-based windows (default: 100 words with 20-word overlap) the default values where based in the `faq_document.txt` provided.
- This ensures:
  - Enough semantic context per chunk.
  - Smooth transitions across document boundaries.
  - A minimum number of meaningful retrieval units.
  - For the default `faq_document.txt` there is also a custom function that breaks the document in topics before the slide window chunking.

### Embedding Generation
- Uses **SentenceTransformers (all-MiniLM-L6-v2)** by default, but others can be used for generating dense embeddings.
- Embeddings are normalized for cosine similarity.
- Raw vectors and metadata are persisted in the `storage/` directory.

### FAISS Vector Index
- A **FAISS IndexFlatIP** index is created using the normalized embeddings.
- This supports fast top-k similarity search via inner product (equivalent to cosine similarity on normalized vectors).
- FAISS index, vectors, and metadata are saved for later reuse.

### Query Pipeline
- For each question:
  1. Embed the query using the same embedding backend.
  2. Retrieve top-k relevant chunks using FAISS.
  3. Construct a context-aware prompt with retrieved chunks.
  4. Generate an answer using an OpenAI-compatible LLM.
  5. Return structured JSON `{ user_question, system_answer, chunks_related }`.

### Evaluation Agent
- Reads all saved samples from `outputs/sample_queries.json`.
- For each entry, passes question, answer, and related chunks into a deterministic evaluation prompt.
- Uses OpenAI to generate a compact evaluation output:
    ```json
    { "score": <0-10>, "reason": "<short explanation>" }
    ```
- Stores all evaluations in outputs/sample_queries_evaluations.json.

## Flow Summary

Below is a simplified representation of the end-to-end pipeline:
```pgsql
                 +-----------------------+
                 |   FAQ Document (.txt) |
                 +-----------+-----------+
                             |
                             v
                [ Chunking & Preprocessing ]
                             |
                             v
                [ Embedding Generation ]
                             |
                             v
                [ FAISS Index Construction ]
                             |
       User question -----> Query Engine -----> Top-k Chunks
                             |                           |
                             |                           v
                             +-----> LLM Generation <----+
                                        |
                                        v
                            Structured JSON Answer
                                        |
                                        v
                           (Optional) Evaluation Agent
```

# Technical Choices

## Chunking Strategy

Sliding window chunking (word-based) was chosen instead of sentence-based splitting because:
- It avoids losing context in very long FAQ sections.
- Overlaps ensure semantically contiguous coverage.
- It is deterministic and easy to tune per document.

## Embedding Strategy

SentenceTransformers is usde by default but it can be switched to OpenAi easily, this choice was due to:
- Strong semantic performance.
- Local embedding without paid API usage.
- Can be changed easily to other embeddigngs.
- Highly efficient and consisten number of dimensions saved as metadata.

## Vector Search

FAISS IndexFlatIP, using inner-product search on normalized vectors (cosine similarity).

Chosen for:
- Very fast approximate search, if the document is not so huge.
- Easily persisted and reused.
- Good trade-off for local running and small projects.

## LLM Answer Generation

- OpenAI-compatible chat.completions API.
- Uses retrieved chunks as context enforcement to reduce hallucination.
- Stores de metadata for chunks used in the prompt.

# Challenges Encountered

## Balancing Chunk Size

- Using too-small chunks degraded search relevance.
- Using too-large chunks reduced precision and made the LLM’s job harder.
- The final 100-word window with 20-word overlap struck a strong balance.

## Prompt Length in Evaluation & Answering

- Large chunks could make prompts exceed token limits.
- Implemented limiting to top k chunks for evaluation and answering.

## Embedding Provider Differences

- Needed a consistent embedding strategy across build and query stages.
- Ensured embeddings for build & query use the same model stored in faiss_meta.json.

## LLM Output Variability

- Ensured evaluator prompts enforce strict JSON output.
- Added fallback parsing logic for malformed responses.

# Potential Improvements

## Hybrid Retrieval

- Combine FAISS dense retrieval with keyword or BM25 sparse retrieval.

- Helps in cases where semantic vectors fail on rare terminology.

## Caching Layer

Cache embeddings for repeated questions to reduce latency and cost.

## Larger or Custom Embedding Models

Upgrade to more powerful embedding models (e.g., text-embedding-3-large) for improved relevance.

## Improved Evaluation Metrics

Incorporate:

- Faithfulness metrics
- Context coverage maps
- Hallucination classification models

## API or UI Layer

Wrap query pipeline into a REST API or chatbot UI for production use.

# Conclusion
The RAG_FAQ_utility provides a structured, maintainable framework for building a scalable FAQ-support system.
It demonstrates the complete lifecycle of a RAG workflow:

Chunk → Embed → Index → Retrieve → Generate → Evaluate.

The architecture is simple enough for rapid iteration while being modular for production hardening. This foundation enables teams to quickly extend into multi-document ingestion, hybrid search approaches, evaluation dashboards, or full integration with customer-facing support chatbots.

The system now serves as a baseline for Retrieval-Augmented Generation capabilities in documentation-rich SaaS environments.