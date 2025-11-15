# src/build_index.py
"""
FAISS-backed index builder for RAG_FAQ_utility.

Outputs:
 - outputs/chunks.json        # list of chunk metadata {"id": int, "text": "..."}
 - outputs/vectors.npy        # raw (float32) vectors in original order
 - outputs/faiss.index        # persisted FAISS index (vectors normalized for cosine)
 - outputs/faiss_meta.json    # small metadata: dim, num_vectors, normalized flag

Usage:
    python src/build_index.py (uses default values for words_chunk_size and overlap_size)
    or 
    python src/build_index.py --input data/faq_document.txt --out_dir outputs --words_chunk_size 100 --overlap_size 20 (to specify different values for words_chunk_size and overlap_size)

Environment:
    OPENAI_API_KEY (optional)  -> if present, OpenAI embeddings will be used
    EMBEDDING_MODEL (optional)-> default: text-embedding-3-small
"""
import sys
sys.dont_write_bytecode = True

import os
import json
import argparse
import numpy as np
from typing import List, Optional
from pathlib import Path
import uuid

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss

import dotenv
dotenv.load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FAQ_FILE = DATA_DIR / "faq_document.txt"
OUTPUTS_DIR = ROOT / "outputs"

JSON_INDENT = 4

def load_text(path: str) -> str:
    """
    Load text from a file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def sliding_window_chunking(text, words_chunk_size, overlap_size):
    """
    Sliding window chunking algorithm to split text into chunks of a given size with a given overlap.
    """

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):

        end = start + words_chunk_size
        chunks.append(' '.join(words[start:end]))
        start += words_chunk_size - overlap_size

    return chunks

# Those values of words_chunk_size and overlap_size are based on the current context of the FAQ document.
def chunk_faq_document(faq_file: Path, words_chunk_size: int = 100, overlap_size: int = 20) -> List[str]:
    """
    Chunk the FAQ document into chunks of a given size with a given overlap.
    """
    faq_text = load_text(faq_file)
    faq_list = faq_text.split('##')[1:]
    docs = []
    for faq in faq_list:
        topic = faq.split('\n')[0]
        body = ''.join(faq.split('\n')[1:])
        chunks = sliding_window_chunking(text=body, words_chunk_size=words_chunk_size, overlap_size=overlap_size)
        for chunk in chunks:
            docs.append({'uuid':str(uuid.uuid4()), 'file':str(faq_file), 'topic':topic, 'text':chunk})

    return docs

def chunk_document(document_path: Path, words_chunk_size: int = 100, overlap_size: int = 20) -> List[str]:
    """
    Chunk the document into chunks of a given size with a given overlap.
    """
    # TODO: Add other document types here
    if document_path == FAQ_FILE:
        docs = chunk_faq_document(faq_file=document_path, words_chunk_size=words_chunk_size, overlap_size=overlap_size)
        docs = [{'id':k, 'uuid':doc['uuid'], 'file':doc['file'], 'topic':doc['topic'], 'text':doc['text']} for k, doc in enumerate(docs)]
    # Fallback to sliding window chunking for other document types
    else:
        text = load_text(document_path)
        chunks = sliding_window_chunking(text=text, words_chunk_size=words_chunk_size, overlap_size=overlap_size)
        docs = [{'id':k, 'uuid':str(uuid.uuid4()), 'file':str(document_path), 'topic':'To be defined', 'text':chunk} for k, chunk in enumerate(chunks)]
    
    return docs



class EmbeddingClient:
    def __init__(self, provider: Optional[str] = None, model_name: Optional[str] = None):
        self.provider = provider or 'sentence-transformers'
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.st = None
        self.openai_client: Optional[OpenAI] = None

        if self.provider == 'sentence-transformers':
            self.model_name = "all-MiniLM-L6-v2"
            self.st = SentenceTransformer(self.model_name)
        else:
            raise RuntimeError("No embedding backend available. Install sentence-transformers.")

    def embed(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Return embedding vectors as nested Python lists. Supports batching."""
        if self.provider == 'openai' and self.openai_client is not None:
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                resp = self.openai_client.embeddings.create(model=self.model_name, input=batch)
                # resp.data is a list matching batch size
                embeddings.extend([d.embedding for d in resp.data])
            return embeddings
        else:
            vecs = self.st.encode(texts, show_progress_bar=True, batch_size=batch_size)
            # sentence-transformers returns numpy arrays
            return [v.tolist() for v in vecs]


def build_index(
    input_path: str = FAQ_FILE,
    out_dir: str = OUTPUTS_DIR,
    words_chunk_size: int = 100,
    overlap_size: int = 20
):
    """
    Build the FAISS index for the input document. 
    It will chunk the document into chunks of a given size with a given overlap, generate embeddings for the chunks, and build the FAISS index.
    It will save the chunks, vectors, and FAISS index to the output directory.
    """
    os.makedirs(out_dir, exist_ok=True)

    docs = chunk_document(document_path=input_path, words_chunk_size=words_chunk_size, overlap_size=overlap_size)
    texts = [d["text"] for d in docs]

    emb = EmbeddingClient()
    print(f"Generating embeddings for {len(texts)} chunks using model={emb.model_name} and provider={emb.provider}")
    vectors = emb.embed(texts)

    # ndarray of shape (N, dim)
    xb = np.array(vectors, dtype="float32")

    # Save chunks and vectors for inspection / fallback
    chunks_path = os.path.join(out_dir, "chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=JSON_INDENT)

    vectors_path = os.path.join(out_dir, "vectors.npy")
    np.save(vectors_path, xb)
    print(f"Saved chunks to {chunks_path} and raw vectors to {vectors_path}")

    # Normalize for cosine similarity and use IndexFlatIP (inner product on normalized vectors)
    dim = xb.shape[1]
    xb_norm = xb.copy()
    faiss.normalize_L2(xb_norm)
    
    # Build FAISS index and persist it
    index = faiss.IndexFlatIP(dim)
    index.add(xb_norm)
    faiss_index_path = os.path.join(out_dir, "faiss.index")
    faiss.write_index(index, faiss_index_path)

    meta = {"provider": emb.provider, "model_name": emb.model_name, "dim": dim, "num_vectors": xb.shape[0], "normalized": True}
    with open(os.path.join(out_dir, "faiss_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=JSON_INDENT)

    print(f"Built FAISS index (dim={dim}, n={xb.shape[0]}) and saved to {faiss_index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/faq_document.txt", help="Path to input text file")
    parser.add_argument("--out_dir", default="outputs", help="Directory to save chunks and faiss index")
    parser.add_argument("--words_chunk_size", type=int, default=100)
    parser.add_argument("--overlap_size", type=int, default=20)
    args = parser.parse_args()

    build_index(
        input_path=args.input,
        out_dir=args.out_dir,
        words_chunk_size=args.words_chunk_size,
        overlap_size=args.overlap_size,
    )
