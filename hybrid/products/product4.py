from __future__ import annotations
import json, os, uuid, torch
from typing import List

from elasticsearch import Elasticsearch
from sentence_transformers import CrossEncoder

# LangChain core
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_core.tools    import tool
from langchain_ollama        import ChatOllama
from langchain.retrievers    import EnsembleRetriever

# Project helpers
from utils.utils                   import load_config, initialize_embeddings, load_faiss_store
from hybrid.custom_sparse_retriever import CustomSparseRetriever
from logger.logger                 import get_logger

logger = get_logger(__name__)
this_dir = os.path.dirname(os.path.abspath(__file__))
cfg = load_config(os.path.join(this_dir, "..", "..", "config.yaml"))

# -----------------------------------------------------------------
# Retrieval TOOL
# -----------------------------------------------------------------
def run_retrieval(query: str, top_k: int = 25) -> str:
    """FAISS + BM25 + cross-encoder re-rank → JSON hits."""
    embeddings = initialize_embeddings(cfg["output_dir"])
    faiss_store = load_faiss_store(cfg["product_store_path"], embeddings)
    es_client = Elasticsearch(cfg["elasticsearch_url"])
    cross_enc = CrossEncoder(cfg["base_cross_encoder"], device=("cuda" if torch.cuda.is_available() else "cpu"))

    dense  = faiss_store.as_retriever(search_kwargs={"k": top_k})
    sparse = CustomSparseRetriever(es_client, cfg["product_index_name"], size=top_k)

    dense_docs = dense.invoke(query)
    sparse_docs = sparse.invoke(query)
    logger.info("Dense %d • Sparse %d", len(dense_docs), len(sparse_docs))

    ensemble = EnsembleRetriever(retrievers=[dense, sparse], weights=[0.5, 0.5])
    docs = ensemble.invoke(query) or []
    if docs:
        scores = cross_enc.predict([(query, d.page_content) for d in docs])
        docs = [d for d,_ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)]

    hits = [{
        "product_id"  : d.metadata.get("product_id"),
        "product_name": d.metadata.get("product_name"),
        "supplier"    : d.metadata.get("Supplier"),
        "city"        : d.metadata.get("city"),
        "country"     : d.metadata.get("country"),
    } for d in docs[:top_k]]

    return json.dumps({"hits": hits})


def pretty_hits(json_str: str) -> str:
    arr = json.loads(json_str).get("hits", [])
    if not arr:
        return "(no hits)"
    return "\n".join(
        f"{i+1:>2}. {h['product_name']} • {h['supplier']} ({h['city']}, {h['country']})"
        for i,h in enumerate(arr)
    )

# -----------------------------------------------------------------
# CLI main (search only)
# -----------------------------------------------------------------

def main() -> None:
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
            k_str = input("  🔢 How many docs? [25] ").strip()
            try:
                k_val = int(k_str) if k_str else 25
            except ValueError:
                print("    Invalid number, using 25.")
                k_val = 25

            # Retrieve and display
            hits_json = run_retrieval.invoke({"query": query, "top_k": k_val})
            print("\n=== Retrieved Products ===")
            print(pretty_hits(hits_json))
            print("==========================\n")

            # Streamed summary
            prompt = f"Summarise in one paragraph:\n{hits_json}"
            print("Assistant: ", end="", flush=True)
            for chunk in llm.stream([HumanMessage(content=prompt)]):
                if hasattr(chunk, "content") and chunk.content:
                    print(chunk.content, end="", flush=True)
            print()
            continue

        print("Unknown command. Use: search | quit")


if __name__ == "__main__":
    main()
