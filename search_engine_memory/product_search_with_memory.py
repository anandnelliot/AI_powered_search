# product2.py — stand-alone LangGraph demo
# ---------------------------------------------------------------
# * Hybrid retrieval: FAISS (dense) + BM25 (Elasticsearch sparse)
# * Cross-encoder re-ranking
# * LangGraph MemorySaver checkpoints (per session_id)
# * Chat history trimmed only for model calls (full history kept)
# * CLI removed; prompts imported externally
# ---------------------------------------------------------------

import os, re
from typing import List, TypedDict

# LangChain core
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, trim_messages

# Hybrid retrieval
from hybrid.custom_sparse_retriever import CustomSparseRetriever
from langchain.retrievers import EnsembleRetriever

# Utilities (project-specific)
from logger.logger import get_logger

# External services
# LangGraph
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# External prompt
from prompts.prompt import custom_recommendation_prompt

logger = get_logger(__name__)

# -----------------------------------------------------------------
# LangGraph memory + message trimmer
# -----------------------------------------------------------------
memory = MemorySaver()
TRIMMER = trim_messages(strategy="last", max_tokens=8, token_counter=len)

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
        dense = faiss_store.as_retriever(search_kwargs={"k": k})
        sparse = CustomSparseRetriever(es_client, bm25_index, size=k)

        _ = dense.invoke(state["question"])
        _ = sparse.invoke(state["question"])
        ensemble_retriever = EnsembleRetriever(
            retrievers=[dense, sparse],
            weights=[0.5, 0.5],
        )
        state["context"] = ensemble_retriever.invoke(state["question"])
    except Exception:
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
def call_llm_recommendation(state: ProductState, llm):
    docs = state.get("context", [])
    if not docs:
        state["llm_text"] = "No documents for recommendation."
        return state

    # Build context snippet
    ctx = "\n".join(
        f"ID: {d.metadata.get('product_id','?')}, Name: {d.metadata.get('product_name','?')}" for d in docs[:25]
    )
    # Format using external prompt
    prompt_str = custom_recommendation_prompt.format(context=ctx, question=state["question"])

    # Trim history and add prompt
    short_history = TRIMMER.invoke(state["messages"])
    response = llm.invoke(short_history + [HumanMessage(content=prompt_str)])

    state["messages"].append(AIMessage(content=response.content))
    state["llm_text"] = response.content
    return state

# -----------------------------------------------------------------
# Build LangGraph pipelines
# -----------------------------------------------------------------
def build_ensemble_pipeline(faiss_store, es, index, cross_enc):
    g = StateGraph(ProductState)
    g.add_node("retrieve", lambda s: retrieve_products(s, faiss_store, es, index))
    g.add_node("rerank", lambda s: cross_encoder_rerank(s, cross_enc))
    g.add_node("metadata", gather_metadata)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "metadata")
    return g.compile(checkpointer=memory)


def build_llm_pipeline(llm):
    g = StateGraph(ProductState)
    g.add_node("llm", lambda s: call_llm_recommendation(s, llm))
    g.add_edge(START, "llm")
    return g.compile(checkpointer=memory)
