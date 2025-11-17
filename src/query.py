# src/query.py
"""
Query pipeline for the RAG_FAQ_utility project.

Features:
 - Loads precomputed chunks.json, vectors.npy, and optionally faiss.index
 - Embeds the user question (OpenAI or sentence-transformers fallback)
 - Performs vector search using FAISS (preferred) or pure numpy
 - Generates an answer using an LLM (if OPENAI_API_KEY set) or extractive fallback
 - Returns JSON: { user_question, system_answer, chunks_related }

Usage:
    python src/query.py --question "How do I invite a user?" --index_path outputs --k 5
"""
import sys
sys.dont_write_bytecode = True

import os
import re
import json
import argparse
import numpy as np
from typing import List, Dict
from pathlib import Path
import time

# Embedding backends
from openai import OpenAI
import faiss

# Local imports
from build_index import EmbeddingClient, ROOT, STORAGE_DIR, JSON_INDENT, CHUNKS_PATH, VECTORS_PATH, FAISS_META_PATH, FAISS_INDEX_PATH

from dotenv import load_dotenv
load_dotenv()

# Constants
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")

OUTPUTS_DIR = ROOT / "outputs"
SAMPLE_OUTPUTS_PATH = OUTPUTS_DIR / "sample_queries.json"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"

def append_json(entry: dict, json_file: Path):
    # Read existing
    if json_file.exists():
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            data = []
    else:
        data = []
    data.append(entry)
    json_file.write_text(json.dumps(data, indent=4), encoding="utf-8")

class QueryEngine:
    def __init__(self, index_path: str = FAISS_INDEX_PATH):
        self.index_path = index_path
        self.tokens_prompt = 0
        self.tokens_completion = 0
        self.total_tokens = 0
        self.model_name = LLM_MODEL

        self.llm_client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL
            )

        if (
            not os.path.exists(index_path) or
            not os.path.exists(CHUNKS_PATH) or 
            not os.path.exists(VECTORS_PATH) or
            not os.path.exists(FAISS_META_PATH) or
            not os.path.exists(FAISS_INDEX_PATH)
            ):
            raise FileNotFoundError(f"Missing {CHUNKS_PATH}, {VECTORS_PATH}, {FAISS_META_PATH}, or {FAISS_INDEX_PATH}. Run `python src/build_index.py` first.")

        # Load faiss_meta.json with metadata about the FAISS index
        with open(FAISS_META_PATH, "r", encoding="utf-8") as file:
            self.faiss_meta = json.load(file)

        # Embedding backend
        self.embedding_client = EmbeddingClient(provider=self.faiss_meta["provider"], model_name=self.faiss_meta["model_name"])

        # Load FAISS index
        self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))

        # Load chunks
        with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
            self.chunks = json.load(file)

        self.vectors = np.load(VECTORS_PATH)  # raw vector matrix (float32)

        # Normalize for cosine similarity (used in numpy fallback)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.norm_vectors = self.vectors / norms


    def embed_query(self, text: str) -> np.ndarray:
        """Embed query text and return normalized vector."""
        if self.faiss_meta["provider"] == "openai":
            resp = self.llm_client.embeddings.create(
                model=self.faiss_meta["model_name"],
                input=text
            )
            embedded = resp.data[0].embedding
        else:
            embedded = self.embedding_client.embed([text])[0]
        
        return np.array(embedded, dtype="float32", ndmin=2)

    def search(self, qvec: np.ndarray, k: int = 3) -> List[Dict]:
        """Return top-k chunks using FAISS or numpy fallback."""
        if self.faiss_index is not None:
            D, I = self.faiss_index.search(qvec, k)
            results = []
            for distance, idx in zip(D[0], I[0]):
                if idx >= 0:
                    results.append({
                        "id": int(idx),
                        "uuid": self.chunks[idx]["uuid"],
                        "topic": self.chunks[idx]["topic"],
                        "distance": float(distance),
                        "text": self.chunks[idx]["text"],
                        "file": self.chunks[idx]["file"]
                    })
            return results

        # Numpy cosine fallback
        sims = (self.norm_vectors @ qvec).tolist()
        idxs = np.argsort(sims)[::-1][:k]
        return [
            {
                "id": int(i),
                "uuid": self.chunks[i]["uuid"],
                "topic": self.chunks[i]["topic"],
                "distance": float(sims[i]),
                "text": self.chunks[i]["text"],
                "file": self.chunks[i]["file"]
            }
            for i in idxs
        ]

    def generate_answer(self, question: str, retrieved: List[Dict]) -> str:
        """Generate an answer using LLM or simple extractive fallback."""

        prompt = "You are a helpful FAQ assistant.\n"
        prompt += "Use ONLY the information provided in the context below to answer. Answer concisely. If unknown, say you don't know.\n"
        prompt += "If the context doesn't contain enough information to fully answer the question, explicitly state what information is missing.\n\n"
        prompt += "CONTEXT: \n"

        for i, ret in enumerate(retrieved):
            
            filename = ret['file'].split('\\')[-1]
            topic = ret['topic']
            prompt += f"SOURCE {i}: {filename} - {topic}\n"
            prompt += f"TEXT: {ret['text']}\n\n"

        resp = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ],
            max_tokens=500,
            temperature=1
        )
        answer = resp.choices[0].message.content.strip()
        
        # Save tokens used for the last query
        self.tokens_prompt = resp.usage.prompt_tokens
        self.tokens_completion = resp.usage.completion_tokens
        self.total_tokens = resp.usage.total_tokens

        return answer

        # Extractive fallback
        if not retrieved:
            return "No relevant information found."

        return (
            f"Based on documentation: {retrieved[0]['text']}\n"
            f"(Using extractive fallback; {len(retrieved)} snippets retrieved.)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--index_path", default=STORAGE_DIR)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    engine = QueryEngine(index_path=args.index_path)
    
    start_time = time.time()
    qvec = engine.embed_query(args.question)
    retrieved = engine.search(qvec, k=args.k)
    end_time = time.time()
    embedding_time = end_time - start_time
    
    start_time = time.time()
    answer = engine.generate_answer(args.question, retrieved)
    end_time = time.time()
    generation_time = end_time - start_time

    out = {
        "user_question": args.question,
        "system_answer": answer,
        "chunks_related": retrieved
    }

    metrics = {
        "question": args.question,
        "answer": answer,
        "model_name": engine.model_name,
        "tokens_prompt": engine.tokens_prompt,
        "tokens_completion": engine.tokens_completion,
        "total_tokens": engine.total_tokens,
        "embedding_time": embedding_time,
        "generation_time": generation_time,
        "k": args.k,
        "retrieved": retrieved,
        "embedding_meta": engine.faiss_meta
    }

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    append_json(out, SAMPLE_OUTPUTS_PATH)
    append_json(metrics, METRICS_PATH)

    print(out)
