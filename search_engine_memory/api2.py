# hybrid/products/api.py — Stateless FastAPI with LangGraph MemorySaver
import uuid
import os
import torch
from typing import Optional, Dict, Any, Generator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from elasticsearch import Elasticsearch
from langchain_core.messages import HumanMessage
from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama

from utils.utils import load_config, initialize_embeddings, load_faiss_store
from search_engine_with_memory.product1 import (
    build_ensemble_pipeline,
    build_recommendation_pipeline,
    build_chat_pipeline,
    ProductState
)
from logger.logger import get_logger
logger = get_logger(__name__)

# Pipelines set in lifespan
ENSEMBLE_PIPELINE: Any = None
RECOMMEND_PIPELINE: Any = None
CHAT_PIPELINE: Any = None

# ——— App & Lifespan —————————————————————————————————————————————————————
def get_lifespan():
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global ENSEMBLE_PIPELINE, RECOMMEND_PIPELINE, CHAT_PIPELINE

        # Load configuration
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(base_dir, '..', 'config.yaml')
        config   = load_config(cfg_path)

        # Prepare retrievers
        device      = 'cuda' if torch.cuda.is_available() else 'cpu'
        cross_enc   = CrossEncoder(config['base_cross_encoder'], device=device)
        embeddings  = initialize_embeddings(config['output_dir'])
        faiss_store = load_faiss_store(config['product_store_path'], embeddings)
        es_client   = Elasticsearch(config['elasticsearch_url'])

        # Build retrieval → rerank → metadata pipeline
        ENSEMBLE_PIPELINE = build_ensemble_pipeline(
            faiss_store,
            es_client,
            config.get('product_index_name', 'products'),
            cross_enc
        )

        # Prepare streaming LLM
        llm = ChatOllama(
            model=config.get('llm_model', 'llama3.1:8b'),
            stream=True,
            temperature=0.1,
            top_k=40,
            repeat_penalty=1.2,
            num_ctx=4096,
            num_predict=1000,
            seed=42
        )

        # Build recommendation-only and chat pipelines
        RECOMMEND_PIPELINE = build_recommendation_pipeline(llm)
        CHAT_PIPELINE      = build_chat_pipeline(
            faiss_store,
            es_client,
            config.get('product_index_name', 'products'),
            cross_enc,
            llm
        )

        logger.info('Pipelines initialized')
        yield

    return lifespan

app = FastAPI(lifespan=get_lifespan())

# ——— Streaming helpers (stateless) ———————————————————————————————————————————

def stream_recommend(session_id: str) -> Generator[str, None, None]:
    # MemorySaver rehydrates state from previous checkpoints
    dummy: Dict[str, Any] = {}
    for token, meta in RECOMMEND_PIPELINE.with_config(thread_id=session_id).stream(
        dummy,
        stream_mode='messages'
    ):
        if meta.get('langgraph_node') == 'recommend' and token.content:
            yield token.content


def stream_chat(session_id: str, message: Optional[str] = None) -> Generator[str, None, None]:
    input_state: Dict[str, Any] = {}
    
    if message and message.strip():
        input_state = {
            'question': message.strip(),
            'messages': [HumanMessage(content=message.strip())]
        }

    for token, meta in CHAT_PIPELINE.with_config(thread_id=session_id).stream(
        input_state,
        stream_mode='messages'
    ):
        if meta.get('langgraph_node') == 'chat' and token.content:
            yield token.content
# ——— Request Models —————————————————————————————————————————————————————

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 200

class RecommendationRequest(BaseModel):
    session_id: str

class ChatRequest(BaseModel):
    session_id: str
    message: Optional[str] = None

# ——— Endpoints ———————————————————————————————————————————————————————

@app.post('/product_search')
async def product_search(req: SearchRequest) -> Dict[str, Any]:
    """
    1) Generate a new session_id
    2) Run retrieval → rerank → metadata
    3) Return session_id + faiss_metadata
    """
    session_id = str(uuid.uuid4())
    initial_state: ProductState = {
        'question':       req.query,
        'k':              req.k,
        'context':        [],
        'faiss_metadata': '',
        'llm_text':       '',
        'loop_step':      0,
        'messages':       [HumanMessage(content=req.query)]
    }

    # Checkpoint initial retrieval state (MemorySaver) and capture updated state
    state: ProductState = ENSEMBLE_PIPELINE.with_config(thread_id=session_id).invoke(initial_state)

    return {
        'session_id':   session_id,
        'product_data': state['faiss_metadata']
    }

@app.post('/product_recommendation')
async def product_recommendation(req: RecommendationRequest):
    """
    Stream recommendations using the recommendation-only pipeline
    """
    return StreamingResponse(
        stream_recommend(req.session_id),
        media_type='text/plain'
    )

@app.post('/product_chat')
async def product_chat(req: ChatRequest):
    return StreamingResponse(
        stream_chat(req.session_id, req.message),
        media_type='text/plain'
    )
