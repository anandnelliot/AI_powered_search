# search_cli.py
from __future__ import annotations
import json
import os
import torch
from typing import List, Any

from elasticsearch import Elasticsearch
from sentence_transformers import CrossEncoder

from langchain_core.documents import Document
from langchain_core.messages  import HumanMessage
from langchain_core.tools     import tool
from langchain.retrievers     import EnsembleRetriever
from langchain_ollama         import ChatOllama

from prompts.prompt import custom_recommendation_prompt
from utils.utils                   import load_config, initialize_embeddings, load_faiss_store
from hybrid.custom_sparse_retriever import CustomSparseRetriever
from logger.logger                  import get_logger

logger = get_logger(__name__)
this_dir = os.path.dirname(os.path.abspath(__file__))
cfg = load_config(os.path.join(this_dir, "..", "..", "config.yaml"))

# ─────────────────────────────────────────────────────────────────
# Module-level placeholders for shared resources
# ─────────────────────────────────────────────────────────────────
faiss_store: Any        = None
es_client:    Elasticsearch = None  # type: ignore
cross_enc:    CrossEncoder    = None  # type: ignore

def deduplicate_documents(docs: List[Document]) -> List[Document]:
    unique, seen = [], set()
    for d in docs:
        pid = d.metadata.get("product_id")
        if pid and pid not in seen:
            unique.append(d)
            seen.add(pid)
    return unique

@tool
def run_retrieval(query: str, top_k: int = 25) -> str:
    """FAISS + BM25 → de-dup → cross-encoder re-rank → JSON hits with full metadata."""
    global faiss_store, es_client, cross_enc

    # 1) dense & sparse retrieval
    dense  = faiss_store.as_retriever(search_kwargs={"k": top_k})
    sparse = CustomSparseRetriever(es_client, cfg["product_index_name"], size=top_k)
    docs   = EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=[0.5, 0.5]
    ).invoke(query) or []

    # 2) de-duplicate
    docs = deduplicate_documents(docs)

    # 3) cross-encoder re-rank
    if docs:
        scores = cross_enc.predict([(query, d.page_content) for d in docs])
        docs   = [d for d,_ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)]

    # 4) emit all desired fields
    hits = [{
        "product_id"  : d.metadata.get("product_id"),
        "product_name": d.metadata.get("product_name"),
        "supplier"    : d.metadata.get("Supplier"),
        "city"        : d.metadata.get("city"),
        "state"       : d.metadata.get("state"),
        "country"     : d.metadata.get("country"),
        "category"    : d.metadata.get("category"),
    } for d in docs[:top_k]]

    return json.dumps({"hits": hits})

def pretty_hits(json_str: str) -> str:
    arr = json.loads(json_str).get("hits", [])
    if not arr:
        return "(no hits)"
    return "\n".join(
        f"{i+1:>2}. [ID: {h.get('product_id','Unknown')}] "
        f"{h.get('product_name','Unknown')} • {h.get('supplier','Unknown')} "
        f"({h.get('city','Unknown')}, {h.get('state','Unknown')}, {h.get('country','Unknown')}) "
        f"[Category: {h.get('category','Unknown')}]"
        for i, h in enumerate(arr)
    )


def main() -> None:
    global faiss_store, es_client, cross_enc

    embeddings   = initialize_embeddings(cfg["output_dir"])
    faiss_store  = load_faiss_store(cfg["product_store_path"], embeddings)
    es_client    = Elasticsearch(cfg["elasticsearch_url"])
    cross_enc    = CrossEncoder(
        cfg["base_cross_encoder"],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    llm = ChatOllama(
        model="llama3.1:8b",
        stream=True,
        temperature=0.2,
        top_k=40,
        repeat_penalty=1.2,
        num_ctx=4096,
        seed=42,
    )

    print("Commands:  search | quit")
    while True:
        cmd = input("\n> ").strip().lower()
        if cmd in {"quit", "exit"}:
            break

        if cmd == "search":
            query = input("  🔍 Query: ").strip()
            try:
                k_val = int(input("  🔢 How many docs? [25] ").strip() or 25)
            except ValueError:
                print("    Invalid input, using 25.")
                k_val = 25

            # 1) Retrieve
            hits_json = run_retrieval.invoke({"query": query, "top_k": k_val})
            print("\n=== Retrieved Products ===")
            print(pretty_hits(hits_json))
            print("==========================\n")

            # 2) Build rich context for prompt
            hits = json.loads(hits_json)["hits"]
            formatted_context = "\n".join(
                "Product ID: {product_id}, Product Name: {product_name}, Supplier: {supplier}, "
                "City: {city}, State: {state}, Country: {country}, Category: {category}".format(**h)
                for h in hits
            )

            # 3) Run LLM with single unified prompt
            prompt = custom_recommendation_prompt.format(
                context=formatted_context,
                question=query
            )
            print("Assistant: ", end="", flush=True)
            for chunk in llm.stream([HumanMessage(content=prompt)]):
                if getattr(chunk, "content", None):
                    print(chunk.content, end="", flush=True)
            print()
            continue

        print("Unknown command. Use: search | quit")


if __name__ == "__main__":
    main()
