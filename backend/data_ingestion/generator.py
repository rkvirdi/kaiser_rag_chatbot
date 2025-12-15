# backend/data_ingestion/generator.py

"""
Simple RAG generation using:

- retriever.retrieve_context(...) for context
- TinyLlama 1.1B Chat as the LLM (CPU-friendly for practice)

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .retriever import retrieve_context


# ------------------------------ CONFIG ---------------------------------------

LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

SYSTEM_PROMPT = """You are an AI assistant for Kaiser Permanente internal staff.
You answer questions strictly based on the provided policy and compliance documents.

Rules:
- Use ONLY the given context to answer.
- If the answer is not clearly in the context, say you are not sure.
- Be concise, accurate, and include dollar amounts, copays, and conditions when relevant.
- Do NOT invent plan names, benefits, or legal language that is not present in the context.
"""


# Stable HF cache inside your project
BASE_DIR = Path(__file__).resolve().parents[2]
HF_CACHE_DIR = BASE_DIR / ".hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Tell HF to use that cache
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
# Optional:
# os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
# os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")


# ---------------------- LOAD MODEL / TOKENIZER ONCE -------------------------

# These will be reused on every call, so the model is not reloaded

_tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL_NAME,
    cache_dir=str(HF_CACHE_DIR),
)
_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    cache_dir=str(HF_CACHE_DIR),
)

_llm = pipeline(
    "text-generation",
    model=_model,
    tokenizer=_tokenizer,
    max_new_tokens=256,   # control only new tokens
    temperature=0.3,
    do_sample=True,
    top_p=0.9,
    truncation=True,     # let tokenizer truncate if still too long
    return_full_text=False,  # 👈 only return new tokens (no prompt echo)
)

# --------------------------- PROMPT BUILDING ---------------------------------


def build_rag_prompt(
    system_prompt: str,
    user_query: str,
    documents: List[str],
    metadatas: List[Dict[str, Any]],
) -> str:
    """
    Build a simple RAG prompt:

    [SYSTEM]
    ...
    [CONTEXT]
    [1] Source: ... etc.
    [USER QUESTION]
    ...
    [ASSISTANT]
    """
    context_blocks = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        src = meta.get("source", "unknown")
        page = meta.get("page", "?")
        context_blocks.append(
            f"[{i}] Source: {src} (page {page})\n{doc}"
        )

    context_str = "\n\n".join(context_blocks)

    prompt = f"""[SYSTEM]
{system_prompt}

[CONTEXT]
The following are relevant excerpts from Kaiser Permanente policy and compliance documents:

{context_str}

[USER QUESTION]
{user_query}

[INSTRUCTIONS]
Using ONLY the information in the [CONTEXT], write a clear and concise answer
for the user. If the answer is not fully supported by the context, say you are
not sure and briefly explain what is missing.

[ASSISTANT]
"""
    return prompt


# -------------------------- GENERATION LOGIC ---------------------------------


def generate_answer(query: str, top_k: int = 2) -> str:
    """
    Practice RAG pipeline:

    1. Retrieve up to top_k chunks from Chroma
    2. Truncate each chunk to keep prompt under model limits
    3. Build prompt with system prompt + context + question
    4. Generate answer with TinyLlama

    Returns:
        A short natural-language answer as a string.
    """
    # If you want to restrict by doc_type, uncomment:
    # where = {"doc_type": "benefit_policy"}
    where: Dict[str, Any] | None = None

    # 1) Retrieve
    results = retrieve_context(query, top_k=top_k, where=where)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        return "I could not find any relevant documents."

    # Limit to top_k docs and trim each to avoid very long prompts
    docs = docs[:top_k]
    metas = metas[:top_k]

    MAX_CHARS_PER_DOC = 1000  # simple character cap per chunk
    docs = [d[:MAX_CHARS_PER_DOC] for d in docs]

    # 2) Build prompt
    prompt = build_rag_prompt(SYSTEM_PROMPT, query, docs, metas)

    # 3) Generate
    output = _llm(prompt, num_return_sequences=1)[0]["generated_text"]

    # Many chat models echo the whole prompt; keep only what comes after [ASSISTANT]
    if "[ASSISTANT]" in output:
        output = output.split("[ASSISTANT]", 1)[-1].strip()

    return output.strip()



# ------------------------------ CLI ENTRYPOINT -------------------------------


def main():
    query = "What is the Ambulance Services copay for Platinum 90 HMO plan?"
    answer = generate_answer(query, top_k=2)
    print(f"\nQ: {query}\n")
    print("A:\n")
    print(answer)


if __name__ == "__main__":
    main()
