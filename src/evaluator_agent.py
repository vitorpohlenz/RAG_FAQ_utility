# src/evaluator_agent.py
"""
OpenAI-based evaluator for RAG_FAQ_utility that considers:
 - user_question (str)
 - system_answer (str)
 - chunks_related (list[dict])  # each dict contains at least 'text' and optional metadata

For each entry in outputs/sample_queries.json the evaluator:
 - sends a concise, deterministic prompt to the LLM that includes the question,
   the system answer, and the retrieved chunks (for grounding)
 - asks the model to return ONLY a JSON object with:
      { "score": <0-10>, "reason": "<brief explanation (max ~60 words)>" }
 - writes results to outputs/sample_queries_evaluations.json preserving the same
   item structure:
      { "user_question": ..., "system_answer": ..., "evaluation": { "score":.., "reason":.. } }

Environment:
  - LLM_API_KEY or OPENAI_API_KEY required
  - LLM_BASE_URL optional (if using a custom OpenAI-compatible host)
  - LLM_MODEL optional (default: "gpt-4o-mini")

Usage:
  python src/evaluator_agent.py
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

load_dotenv()

# OpenAI client import (matching other repo files)
try:
    from openai import OpenAI
except Exception as e:
    print("Missing OpenAI Python SDK. Install `openai` package.", file=sys.stderr)
    raise

# Config from env
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")  # optional
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

if not LLM_API_KEY:
    print("LLM_API_KEY or OPENAI_API_KEY environment variable is required.", file=sys.stderr)
    sys.exit(1)

# Paths
ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "outputs" / "sample_queries.json"
OUTPUT_PATH = ROOT / "outputs" / "queries_evaluations.json"

# Initialize client
llm = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL
)


def build_prompt(question: str, answer: str, chunks: List[Dict[str, Any]], max_chunks: int = 10) -> Dict[str, str]:
    """
    Construct a deterministic prompt asking the model to produce ONLY a JSON object.
    Included `max_chunks` for grounding.
    Returns a dict with keys "system" and "user" representing messages.
    """
    system = (
        """You are an objective evaluator specialized in assessing FAQ answers.
        Given a USER_QUESTION, a proposed SYSTEM_ANSWER, and retrieved context chunks, 
        you must judge how well the answer addresses the question and whether it is 
        grounded in the provided context. Produce ONLY a JSON object with two keys
          - score: integer between 0 and 10 (10 = best)
          - reason: short (<= 60 words) explanation for the score
        Do NOT output any other text or commentary. Be concise and deterministic. If not sure, return a score of -1 and reason 'Not sure'."""
    )

    # Build a concise user message with question, answer and the top chunks
    user_lines = []
    user_lines.append("USER_QUESTION:")
    user_lines.append(question.strip())
    user_lines.append("")
    user_lines.append("SYSTEM_ANSWER:")
    user_lines.append(answer.strip() or "<empty>")
    user_lines.append("")
    user_lines.append(f"RETRIEVED CHUNKS (up to {max_chunks}):")

    for i, c in enumerate(chunks[:max_chunks]):
        # include minimal metadata for context (topic and file) if present
        meta = []
        if isinstance(c, dict):
            topic = c.get("topic")
            if topic:
                meta.append(f"topic={topic.strip()}")
            fid = c.get("file")
            if fid:
                meta.append(f"file={Path(fid).name}")
        meta_str = (" [" + ", ".join(meta) + "]") if meta else ""
        text = c.get("text") if isinstance(c, dict) else str(c)
        # truncate chunk text to a reasonable length to keep prompt size bounded
        text_trim = text.strip().replace("\n", " ")
        if len(text_trim) > 800:
            text_trim = text_trim[:800].rstrip() + " ..."

        user_lines.append(f"CHUNK {i+1}{meta_str}: {text_trim}")

    user_lines.append("")

    return {"system": system, "user": "\n".join(user_lines)}


def safe_parse_json(text: str) -> Dict[str, Any]:
    """
    Try to extract and parse the first JSON object in the given text.
    If parsing fails, return a fallback object with score 0 and the raw output as reason.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # find first { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
    return {"score": 0, "reason": f"Could not parse model output. Raw: {text[:300]}"}


def evaluate_item(item: Dict[str, Any], max_chunks: int = 5) -> Dict[str, Any]:
    """
    Evaluate a single item using the LLM. Returns a dict:
      { "user_question": ..., "system_answer": ..., "evaluation": {"score":.., "reason":..} }
    """
    question = item.get("user_question")
    answer = item.get("system_answer")
    chunks = item.get("chunks_related")

    prompt = build_prompt(question=question, answer=answer, chunks=chunks, max_chunks=max_chunks)

    try:
        resp = llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            max_tokens=200,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        return {"user_question": question, "system_answer": answer, "evaluation": {"score": 0, "reason": f"LLM error: {e}"}}

    # extract assistant text (SDK variations)
    try:
        content = resp.choices[0].message.content.strip()
    except Exception:
        content = getattr(resp, "output_text", str(resp))

    parsed = safe_parse_json(content)
    score = parsed.get("score", 0)
    reason = parsed.get("reason", "") or parsed.get("explanation", "")

    # normalize score to int in 0..10
    try:
        score = int(score)
    except Exception:
        try:
            score = int(float(score))
        except Exception:
            score = 0
    score = max(0, min(10, score))

    evaluation = {"score": score, "reason": reason}
    return {"user_question": question, "system_answer": answer, "evaluation": evaluation}


def evaluate_all():
    if not INPUT_PATH.exists():
        print(f"Input file not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    try:
        items = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            raise ValueError("sample_queries.json must contain a top-level JSON list")
    except Exception as e:
        print(f"Failed to read/parse {INPUT_PATH}: {e}", file=sys.stderr)
        sys.exit(1)

    results: List[Dict[str, Any]] = []
    for item in items:
        out = evaluate_item(item, max_chunks=5)
        results.append(out)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Evaluated {len(results)} items. Results written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    evaluate_all()
