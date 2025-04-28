import uuid
import os
import asyncio
from typing import Dict, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from sentence_transformers import CrossEncoder
from elasticsearch import Elasticsearch
from langchain_ollama import ChatOllama

from hybrid.products.product2 import (
    ProductState,
    build_ensemble_pipeline,
    build_llm_pipeline,
    memory,
)
from utils.utils import load_config, initialize_embeddings, load_faiss_store
from logger.logger import get_logger

logger = get_logger(__name__)

ENSEMBLE_PIPELINE = None
LLM_PIPELINE = None
SESSION_STORE: Dict[str, ProductState] = {}

_SEARCH_KEYS = {"find", "show", "product", "supplier", "search", "near", "laptop", "engine", "oil"}

def needs_retrieval(text: str) -> bool:
    return any(w in text.lower() for w in _SEARCH_KEYS)

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 200

class FollowUpRequest(BaseModel):
    session_id: str
    message: str

app = FastAPI()

async def _init_pipelines():
    global ENSEMBLE_PIPELINE, LLM_PIPELINE

    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(base_dir,"..","..", "config.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cross_encoder = CrossEncoder(cfg["base_cross_encoder"], device=device)
    embeddings = initialize_embeddings(cfg["output_dir"])
    faiss_store = load_faiss_store(cfg["product_store_path"], embeddings)
    es_client = Elasticsearch(cfg["elasticsearch_url"])

    llm = ChatOllama(
        model="llama3.1:8b",
        stream=True,
        temperature=0.1,
        top_k=40,
        repeat_penalty=1.2,
        num_ctx=4096,
        num_predict=1000,
        seed=42,
    )

    ENSEMBLE_PIPELINE = build_ensemble_pipeline(
        faiss_store=faiss_store,
        es_client=es_client,
        bm25_index=cfg.get("product_index_name", "products"),
        cross_encoder=cross_encoder,
    )
    LLM_PIPELINE = build_llm_pipeline(llm)

@app.on_event("startup")
async def _on_start():
    await _init_pipelines()

@app.post("/product_search")
async def product_search(req: SearchRequest):
    if ENSEMBLE_PIPELINE is None:
        raise HTTPException(500, "Pipelines not ready")

    sid = str(uuid.uuid4())
    state: ProductState = {
        "question": req.query,
        "k": req.k,
        "context": [],
        "faiss_metadata": "",
        "llm_text": "",
        "loop_step": 0,
        "messages": [HumanMessage(content=req.query)],
    }

    state = await asyncio.to_thread(
        ENSEMBLE_PIPELINE.with_config(thread_id=sid).invoke, state
    )

    SESSION_STORE[sid] = state
    return {"session_id": sid, "product_data": state["faiss_metadata"]}

@app.post("/chat_followup")
async def chat_followup(req: FollowUpRequest):
    if LLM_PIPELINE is None:
        raise HTTPException(500, "Pipelines not ready")

    state = SESSION_STORE.get(req.session_id)
    if state is None:
        raise HTTPException(400, "Unknown session_id")

    state["messages"].append(HumanMessage(content=req.message))
    state["question"] = req.message
    state["loop_step"] += 1

    if needs_retrieval(req.message):
        state = await asyncio.to_thread(
            ENSEMBLE_PIPELINE.with_config(thread_id=req.session_id).invoke, state
        )

    async def gen():
        async for msg, meta in LLM_PIPELINE.with_config(thread_id=req.session_id).astream(
            state, stream_mode="messages"
        ):
            if meta["langgraph_node"] == "llm" and msg.content:
                yield msg.content
        SESSION_STORE[req.session_id] = state

    return StreamingResponse(gen(), media_type="text/plain")
