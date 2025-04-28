import os
import torch
import mlflow
import numpy as np
from sentence_transformers import SentenceTransformer

# If you use a logger in your project, import it here
# Otherwise, you can just use print statements or Python's logging
from logger.logger import get_logger
logger = get_logger(__file__)

# ----------------------------
# 1) Initialize Model from MLflow
# ----------------------------
def initialize_mlflow_model(model_uri: str) -> SentenceTransformer:
    """
    Loads a SentenceTransformer model that was logged via MLflow's sentence_transformers flavor.
    Moves it to GPU if available.
    """
    try:
        # Point MLflow to your tracking server
        mlflow.set_tracking_uri("http://192.168.1.227:5000")

        # Load the SentenceTransformer model from MLflow
        model = mlflow.sentence_transformers.load_model(model_uri)

        # Move to GPU or stay on CPU, as appropriate
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        logger.info(f"Model loaded from MLflow: {model_uri}, using device: {device}")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize model from {model_uri}: {e}", exc_info=True)
        raise e

# ----------------------------
# 2) Create a Custom Embeddings Wrapper for LangChain
# ----------------------------
from langchain.embeddings.base import Embeddings

class MLflowSentenceTransformerEmbeddings(Embeddings):
    """
    A simple wrapper that lets us use a raw SentenceTransformer model
    (loaded from MLflow or otherwise) with LangChain's vector stores.
    """
    def __init__(self, st_model: SentenceTransformer):
        self.model = st_model
        # For best performance, you may want to store the device here as well:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple documents (list of strings) and return a list of embeddings,
        where each embedding is a list of floats.
        """
        # Sentence-Transformers .encode() handles batching automatically
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text and return its embedding as a list of floats.
        """
        embedding = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
        return embedding.tolist()

# ----------------------------
# 3) Example Usage
# ----------------------------
if __name__ == "__main__":
    # Example model URI (registered model "jsearch_model", version 12):
    model_uri = "models:/jsearch_model/42"
    # or "models:/jsearch_model/staging" if you have a stage named 'staging'

    # A) Load the model via MLflow
    st_model = initialize_mlflow_model(model_uri)

    # B) Verify basic encoding works
    sample_embedding = st_model.encode(["Hello from MLflow & SentenceTransformers!"])
    print("Raw embedding from the model:", sample_embedding)

    # C) Wrap in our custom embeddings class for LangChain
    from langchain_community.vectorstores import FAISS
    from langchain.docstore.document import Document


    embedding_function = MLflowSentenceTransformerEmbeddings(st_model)

    # D) Create some dummy texts
    texts = [
        "LangChain makes it easy to build LLM-powered apps.",
        "MLflow helps you version and serve your machine learning models.",
        "FAISS is a library for efficient similarity search."
    ]

    # E) Build a FAISS vector store from text
    #    Each text is turned into a Document for demonstration
    docs = [Document(page_content=t) for t in texts]

    vector_store = FAISS.from_documents(
        docs, 
        embedding_function
    )

    # F) Test a query
    query = "What helps with versioning ML models?"
    found_docs = vector_store.similarity_search(query, k=2)
    print("\n=== Search Results ===")
    for i, doc in enumerate(found_docs, start=1):
        print(f"Result #{i}: {doc.page_content}")
