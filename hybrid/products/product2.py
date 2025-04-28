# product2.py — stand-alone LangGraph demo
# ---------------------------------------------------------------
# * Hybrid retrieval: FAISS (dense) + BM25 (Elasticsearch sparse)
# * Cross-encoder re-ranking
# * LangGraph MemorySaver checkpoints (per session_id)
# * Chat history trimmed only for model calls (full history kept)
# * CLI with top-level commands:
#       search   → new session, new query
#       chat     → enter chat sub-prompt
#       exit     → quit
#   Inside the chat sub-prompt:
#       new product <query>  → pull fresh docs in the same session
#       back / <empty line>  → leave chat
# ---------------------------------------------------------------

import os, re
from typing import List, TypedDict

# LangChain core
from langchain_core.prompts   import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages  import HumanMessage, AIMessage, BaseMessage, trim_messages
from langchain_ollama         import ChatOllama

# Hybrid retrieval
from hybrid.custom_sparse_retriever import CustomSparseRetriever
from langchain.retrievers           import EnsembleRetriever

# Utilities (project-specific)
from utils.utils  import load_config, initialize_embeddings, load_faiss_store
from logger.logger import get_logger

# External services
from elasticsearch            import Elasticsearch
import torch, uuid
from sentence_transformers    import CrossEncoder

# LangGraph
from langgraph.graph          import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

logger = get_logger(__name__)

# -----------------------------------------------------------------
# LangGraph memory + message trimmer
# -----------------------------------------------------------------
memory   = MemorySaver()
TRIMMER  = trim_messages(strategy="last", max_tokens=8, token_counter=len)

# -----------------------------------------------------------------
# ProductState schema
# -----------------------------------------------------------------
class ProductState(TypedDict):
    question: str
    k: int
    context: List[Document]
    faiss_metadata: str
    llm_text: str
    loop_step: int
    messages: List[BaseMessage]

# -----------------------------------------------------------------
# Retrieval helpers
# -----------------------------------------------------------------
def retrieve_products(state: ProductState, faiss_store, es_client, bm25_index):
    """Dense (FAISS) + sparse (BM25) ensemble retrieval."""
    k = state.get("k", 200)
    try:
        dense  = faiss_store.as_retriever(search_kwargs={"k": k})
        sparse = CustomSparseRetriever(es_client, bm25_index, size=k)

        dense_docs  = dense.invoke(state["question"])
        sparse_docs = sparse.invoke(state["question"])
        logger.info("Dense retrieved %d docs • Sparse %d docs", len(dense_docs), len(sparse_docs))

        ensemble_retriever = EnsembleRetriever(
    retrievers=[dense, sparse],
    weights=[0.5, 0.5],
)
        state["context"] = ensemble_retriever.invoke(state["question"])
    except Exception as e:
        logger.exception("Retrieval failure: %s", e)
        state["context"] = []
    return state


def cross_encoder_rerank(state: ProductState, cross_encoder):
    docs = state.get("context", [])
    if not docs:
        return state
    scores = cross_encoder.predict([(state["question"], d.page_content) for d in docs])
    state["context"] = [d for d, _ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)]
    return state


def _dedup(docs: List[Document]) -> List[Document]:
    seen, unique = set(), []
    for d in docs:
        pid = d.metadata.get("product_id")
        if pid and pid not in seen:
            unique.append(d)
            seen.add(pid)
    return unique


def gather_metadata(state: ProductState):
    docs = _dedup(state.get("context", []))
    if not docs:
        state["faiss_metadata"] = "No documents retrieved; no metadata."
    else:
        lines = [", ".join(f"{k}: {v}" for k, v in d.metadata.items()) for d in docs]
        state["faiss_metadata"] = "\n".join(lines)
    return state

# -----------------------------------------------------------------
# LLM recommendation
# -----------------------------------------------------------------
_PROMPT = PromptTemplate(
    template=(
        "You are an intelligent assistant that summarizes the product(s) "
        "based on the user's query from the available products.\n\n"
        "**Instructions:**\n"
        "- Give a concise paragraph summary integrating product names & locations.\n"
        "- Suggest related products **from the context only**.\n"
        "- If the query is vague, gently ask for clarification.\n"
        "- Encourage follow-up questions.\n\n"
        "Context:\n{context}\n\nUser Query:\n{question}\n\nRecommendation:\n"
    ),
    input_variables=["context", "question"],
)

def call_llm_recommendation(state: ProductState, llm):
    docs = state.get("context", [])
    if not docs:
        state["llm_text"] = "No documents for recommendation."
        return state

    ctx = "\n".join(
        f"Product ID: {d.metadata.get('product_id','?')}, "
        f"Product Name: {d.metadata.get('product_name','?')}, "
        f"Supplier: {d.metadata.get('Supplier','?')}, "
        f"City: {d.metadata.get('city','?')}, State: {d.metadata.get('state','?')}, "
        f"Country: {d.metadata.get('country','?')}, Category: {d.metadata.get('category','?')}"
        for d in docs[:25]
    )
    prompt = _PROMPT.format(context=ctx, question=state["question"])

    short_history = TRIMMER.invoke(state["messages"])
    response = llm.invoke(short_history + [HumanMessage(content=prompt)])

    state["messages"].append(AIMessage(content=response.content))
    state["llm_text"] = response.content
    return state

# -----------------------------------------------------------------
# Build LangGraph pipelines
# -----------------------------------------------------------------
def build_ensemble_pipeline(faiss_store, es, index, cross_enc):
    g = StateGraph(ProductState)
    g.add_node("retrieve", lambda s: retrieve_products(s, faiss_store, es, index))
    g.add_node("rerank",   lambda s: cross_encoder_rerank(s, cross_enc))
    g.add_node("metadata", gather_metadata)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank",   "metadata")
    return g.compile(checkpointer=memory)

def build_llm_pipeline(llm):
    g = StateGraph(ProductState)
    g.add_node("llm", lambda s: call_llm_recommendation(s, llm))
    g.add_edge(START, "llm")
    return g.compile(checkpointer=memory)

# -----------------------------------------------------------------
# CLI main
# -----------------------------------------------------------------
def main():
    # ---- configuration & components ---------------------------------
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    llm = ChatOllama(
        model="llama3.1:8b", stream=True, temperature=0.1,
        top_k=40, repeat_penalty=1.2, num_ctx=4096, num_predict=1000, seed=42,
    )

    embeddings  = initialize_embeddings(cfg["output_dir"])
    faiss_store = load_faiss_store(cfg["product_store_path"], embeddings)
    es_client   = Elasticsearch(cfg["elasticsearch_url"])
    cross_enc   = CrossEncoder(cfg["base_cross_encoder"], device=device)

    ensemble_pipe = build_ensemble_pipeline(
        faiss_store, es_client, cfg.get("product_index_name", "products"), cross_enc
    )
    llm_pipe = build_llm_pipeline(llm)

    # ---- session variables ------------------------------------------
    state: ProductState | None = None
    session_id: str | None = None

    # ---- top-level CLI loop -----------------------------------------
    print("Type  search  |  chat  |  exit")
    while True:
        cmd = input("\n> ").strip().lower()

        # ---------- exit ---------------------------------------------
        if cmd in {"exit", "quit"}:
            break

        # ---------- search (new session) -----------------------------
        if cmd == "search":
            query = input("  🔍 Query: ").strip()
            k_str = input("  🔢 How many docs? [200] ").strip()
            try:
                k_val = int(k_str) if k_str else 200
            except ValueError:
                print("    Invalid number, using 200.")
                k_val = 200

            session_id = str(uuid.uuid4())
            state = {
                "question": query,
                "k": k_val,
                "context": [],
                "faiss_metadata": "",
                "llm_text": "",
                "loop_step": 0,
                "messages": [HumanMessage(content=query)],
            }

            state = ensemble_pipe.with_config(thread_id=session_id).invoke(state)

            print("\n=== Retrieved Products (deduped) ===")
            print(state["faiss_metadata"] or "(no hits)")
            print("====================================\n")

            print("Assistant: ", end="", flush=True)
            for msg, meta in llm_pipe.with_config(thread_id=session_id).stream(
                state, stream_mode="messages"
            ):
                if meta["langgraph_node"] == "llm" and msg.content:
                    print(msg.content, end="", flush=True)
            print()
            continue

        # ---------- chat sub-prompt ----------------------------------
        if cmd == "chat":
            if state is None:
                print("  ⚠️  Run a search first.")
                continue

            print("Entering chat.  (sub-commands inside chat: new product  |  back)")
            while True:
                sub = input("chat> ").strip()
                if sub.lower() in {"back", ""}:
                    print("Leaving chat.")
                    break

                # ----- new product retrieval (same session) ----------
                if sub.lower().startswith("new product"):
                    query = sub[len("new product"):].strip() \
                            or input("  🔍 Query: ").strip()

                    state["question"] = query
                    state["k"]        = 25           # fixed k
                    state["messages"].append(HumanMessage(content=query))

                    print("  ↻ Pulling fresh documents …")
                    state = ensemble_pipe.with_config(thread_id=session_id).invoke(state)

                    # assistant reply based on new context
                    print("Assistant: ", end="", flush=True)
                    for msg, meta in llm_pipe.with_config(thread_id=session_id).stream(
                        state, stream_mode="messages"
                    ):
                        if meta["langgraph_node"] == "llm" and msg.content:
                            print(msg.content, end="", flush=True)
                    print()
                    continue  # stay inside chat loop

                # ----- normal follow-up -----------------------------
                state["question"] = sub
                state["messages"].append(HumanMessage(content=sub))

                print("Assistant: ", end="", flush=True)
                for msg, meta in llm_pipe.with_config(thread_id=session_id).stream(
                    state, stream_mode="messages"
                ):
                    if meta["langgraph_node"] == "llm" and msg.content:
                        print(msg.content, end="", flush=True)
                print()
            continue

        # ---------- unknown command ----------------------------------
        print("  Unknown command. Type  search  |  chat  |  exit")


if __name__ == "__main__":
    main()
