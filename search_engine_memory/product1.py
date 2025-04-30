# product2.py — stand-alone LangGraph demo with 3 pipelines
# ---------------------------------------------------------------
# * Hybrid retrieval: FAISS (dense) + BM25 (Elasticsearch sparse)
# * Cross-encoder re-ranking
# * Three LangGraph pipelines:
#     1) retrieve → rerank → metadata
#     2) llm recommendation only
#     3) retrieve → rerank → metadata → chat
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

# LangGraph
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# External prompts
from prompts.prompt import custom_recommendation_prompt, custom_chat_prompt

# Logging
from logger.logger import get_logger
logger = get_logger(__name__)

# -----------------------------------------------------------------
# Shared utilities
# -----------------------------------------------------------------
memory  = MemorySaver()
TRIMMER = trim_messages(strategy="last", max_tokens=8, token_counter=len)

class ProductState(TypedDict):
    question:       str
    k:              int
    context:        List[Document]
    faiss_metadata: str
    llm_text:       str
    loop_step:      int
    messages:       List[BaseMessage]

def retrieve_products(state: ProductState, faiss_store, es_client, bm25_index: str) -> ProductState:
    k = state.get("k", 200)
    try:
        dense  = faiss_store.as_retriever(search_kwargs={"k": k})
        sparse = CustomSparseRetriever(es_client, bm25_index, size=k)
        ens    = EnsembleRetriever(retrievers=[dense, sparse], weights=[0.5, 0.5])
        state["context"] = ens.invoke(state["question"])
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        state["context"] = []
    return state

def cross_encoder_rerank(state: ProductState, cross_encoder) -> ProductState:
    docs = state.get("context", [])
    if not docs:
        return state
    try:
        pairs  = [(state["question"], d.page_content) for d in docs]
        scores = cross_encoder.predict(pairs)
        state["context"] = [d for d,_ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)]
    except Exception as e:
        logger.error(f"Rerank error: {e}")
    return state

def _dedup(docs: List[Document]) -> List[Document]:
    seen, out = set(), []
    for d in docs:
        pid = d.metadata.get("product_id")
        if pid and pid not in seen:
            out.append(d); seen.add(pid)
    return out

def gather_metadata(state: ProductState) -> ProductState:
    docs = _dedup(state.get("context", []))
    if not docs:
        state["faiss_metadata"] = "No documents retrieved; no metadata."
    else:
        lines = [", ".join(f"{k}: {v}" for k,v in d.metadata.items()) for d in docs]
        state["faiss_metadata"] = "\n".join(lines)
    return state

# -----------------------------------------------------------------
# 1) Retrieval‐only pipeline
# -----------------------------------------------------------------
def build_ensemble_pipeline(faiss_store, es_client, index: str, cross_encoder):
    g = StateGraph(ProductState)
    g.add_node("retrieve", lambda s: retrieve_products(s, faiss_store, es_client, index))
    g.add_node("rerank",   lambda s: cross_encoder_rerank(s, cross_encoder))
    g.add_node("metadata", gather_metadata)
    g.add_edge(START,      "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank",   "metadata")
    return g.compile(checkpointer=memory)

# -----------------------------------------------------------------
# 2) Recommendation‐only pipeline
# -----------------------------------------------------------------
def call_llm_recommendation(state: ProductState, llm) -> ProductState:
    docs = state.get("context", [])
    if not docs:
        state["llm_text"] = "No documents for recommendation."
        return state
    top25 = docs[:25]
    ctx = "\n".join(
        f"Product ID: {doc.metadata.get('product_id', 'Unknown')}, "
        f"Product Name: {doc.metadata.get('product_name', 'Unknown')}, "
        f"Supplier: {doc.metadata.get('Supplier', 'Unknown')}, "
        f"City: {doc.metadata.get('city', 'Unknown')}, "
        f"State: {doc.metadata.get('state', 'Unknown')}, "
        f"Country: {doc.metadata.get('country', 'Unknown')}, "
        f"Category: {doc.metadata.get('category', 'Unknown')}"
        for doc in top25
    )
    
    prompt = custom_recommendation_prompt.format(context=ctx, question=state["question"])
    history = TRIMMER.invoke(state["messages"])
    resp = llm.invoke(history + [HumanMessage(content=prompt)])
    state["messages"].append(AIMessage(content=resp.content))
    state["llm_text"] = resp.content
    return state

def build_recommendation_pipeline(llm):
    g = StateGraph(ProductState)
    g.add_node("recommend", lambda s: call_llm_recommendation(s, llm))
    g.add_edge(START, "recommend")
    return g.compile(checkpointer=memory)

# -----------------------------------------------------------------
# 3) Chat pipeline (with retrieval included)
# -----------------------------------------------------------------
def call_llm_chat(state: ProductState, llm) -> ProductState:
    # 1) History trimming
    history = TRIMMER.invoke(state["messages"])
    # Map message classes to speaker labels
    hist_str = "\n".join(
        f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in history
    )

    # 2) Build context block from top retrieved docs
    top_docs = state.get("context", [])[:25]
    ctx_block = "\n".join(
        f"Product ID: {doc.metadata.get('product_id', 'Unknown')}, "
        f"Product Name: {doc.metadata.get('product_name', 'Unknown')}, "
        f"Supplier: {doc.metadata.get('Supplier', 'Unknown')}, "
        f"City: {doc.metadata.get('city', 'Unknown')}, "
        f"State: {doc.metadata.get('state', 'Unknown')}, "
        f"Country: {doc.metadata.get('country', 'Unknown')}, "
        f"Category: {doc.metadata.get('category', 'Unknown')}"
        for doc in top_docs
    )

    # 3) Format prompt with both context and history
    prompt = custom_chat_prompt.format(
        context=ctx_block,
        history=hist_str,
        question=state["question"],
    )

    # 4) Invoke the LLM
    resp = llm.invoke([HumanMessage(content=prompt)])

    # 5) Update state
    state["messages"].append(AIMessage(content=resp.content))
    state["llm_text"] = resp.content
    return state


def build_chat_pipeline(faiss_store, es_client, index: str, cross_encoder, llm):
    g = StateGraph(ProductState)
    g.add_node("retrieve",   lambda s: retrieve_products(s, faiss_store, es_client, index))
    g.add_node("rerank",     lambda s: cross_encoder_rerank(s, cross_encoder))
    g.add_node("metadata",   gather_metadata)
    g.add_node("chat",       lambda s: call_llm_chat(s, llm))
    g.add_edge(START,          "retrieve")
    g.add_edge("retrieve",   "rerank")
    g.add_edge("rerank",     "metadata")
    g.add_edge("metadata",   "chat")
    return g.compile(checkpointer=memory)
