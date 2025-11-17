# tests/test_core.py
import sys
sys.dont_write_bytecode = True

import os
from pathlib import Path
import numpy as np
import re

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.append(str(SRC_DIR))

from build_index import STORAGE_DIR, CHUNKS_PATH, VECTORS_PATH, FAISS_META_PATH, FAISS_INDEX_PATH
from query import QueryEngine


def test_build():
    """Test the build index."""
    # Required artifacts
    assert os.path.exists(STORAGE_DIR), "storage directory missing, run `python src/build_index.py` first"
    assert os.path.exists(CHUNKS_PATH), "chunks.json missing"
    assert os.path.exists(VECTORS_PATH), "vectors.npy missing"
    assert os.path.exists(FAISS_META_PATH), "faiss_meta.json missing"
    assert os.path.exists(FAISS_INDEX_PATH), "faiss.index missing"

    # Check if the faiss index is not empty
    if os.path.exists(FAISS_INDEX_PATH):
        assert os.path.getsize(FAISS_INDEX_PATH) > 0, "faiss.index exists but is empty"

def test_query_engine(k: int = 3):
    """Test the query engine."""
    
    query_engine = QueryEngine()
    question = "What is RBAC?"
    qvec = query_engine.embed_query(question)
    retrieved = query_engine.search(qvec, k=k)
    answer = query_engine.generate_answer(question, retrieved)
    
    assert len(retrieved) == k, "Number of retrieved chunks is not equal to k"
    assert np.any(['RBAC' in ret['text'] for ret in retrieved]), "Answer does not contain RBAC"
    assert answer is not None, "Answer is None"
    assert 'role based access control' in re.sub(r'[^\w\s]', ' ', answer).lower(), "Answer does not contain role based access control"
    
