# query.py
import json
import subprocess
import time

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_HOST = "http://localhost:6333"
HF_API_TOKEN = "HF_API_TOKEN"
HF_MODEL = "gpt2"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(url=QDRANT_HOST)
COLLECTION_NAME = "my_docs"


def call_ollama_phi3(prompt: str, timeout: int = 60) -> str:
    """
    Send prompt to local ollama phi3 and return text output.
    Uses subprocess to run: echo "prompt" | ollama run phi3
    """
    # Use shell-safe encoding via bytes; avoid shell=True for safety.
    try:
        proc = subprocess.run(
            ["ollama", "run", "phi3"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,  # we'll handle error reporting below
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Ollama invocation timed out after {timeout}s") from e

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama returned non-zero exit code {proc.returncode}: {stderr}"
        )

    # Ollama prints the model text to stdout. Decode and return.
    text = proc.stdout.decode("utf-8", errors="replace").strip()
    return text


def retrieve(query, top_k=5):
    q_emb = embedder.encode([query])[0].tolist()
    hits = client.search(  # type: ignore
        collection_name=COLLECTION_NAME, query_vector=q_emb, limit=top_k
    )
    # hits: list with .payload and .score
    contexts = []
    for h in hits:
        ctx = h.payload.get("text")
        contexts.append({"text": ctx, "score": h.score})
    return contexts


if __name__ == "__main__":
    user_query = "Describe me"
    # retrieve() should return list of dicts with key 'text' as before
    candidates = retrieve(user_query, top_k=8)
    candidates = sorted(
        [c for c in candidates if c.get("text")],
        key=lambda x: x.get("score", 0),
        reverse=True,
    )[:8]
    if not candidates:
        print("No context found; please ingest profile documents first.")
        raise SystemExit(1)

    # Build context text (you can include IDs/score if available)
    ctx_text = "\n\n".join(
        [
            f"[chunk={i} score={c.get('score',0):.3f}] {c['text']}"
            for i, c in enumerate(candidates)
        ]
    )

    # Prompt template (concise + grounding instruction)
    system_prompt = (
        "You are an assistant. Using ONLY the CONTEXT below, write a short first-person professional summary.\n"
        "Follow these rules strictly:\n"
        "1. If the CONTEXT contains a sentence starting with 'My name is', extract the name exactly as written "
        "and start the answer with: 'I am <name>'.\n"
        "2. If no such sentence exists, start the answer with: 'I am a backend developer.'\n"
        "3. DO NOT invent names, years of experience, company names, or facts not present in CONTEXT.\n"
        "4. Use ELABORATIVE sentences (10 minimum).\n"
        "5. Do not paraphrase any name — use it exactly as it appears.\n\n"
        f"CONTEXT:\n{ctx_text}\n\nAnswer:"
    )

    # Call local phi3 via ollama
    start = time.time()
    try:
        model_output = call_ollama_phi3(system_prompt, timeout=120)
    except RuntimeError as e:
        print("Error calling phi3:", str(e))
        raise

    latency = time.time() - start

    # Print nicely
    result = {
        "model": "phi3 (ollama)",
        "latency_seconds": latency,
        "prompt_length_chars": len(system_prompt),
        "response": model_output,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
