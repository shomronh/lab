from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
import uvicorn
from typing import List
import os
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

# Set cache directory to match Docker volume mount path
os.environ["TRANSFORMERS_CACHE"] = "/models"

app = FastAPI(title="Embeddings API")

# Track if model is ready
model_initialized = False

# Initialize embeddings with cache configuration
try:
    # HuggingFaceEmbeddings will automatically handle model downloading and caching
    embeddings = HuggingFaceEmbeddings(
        # all-MiniLM-L6-v2 is a smaller, faster model suitable for CPU inference
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        # matches our volume mount
        cache_folder="/models",  
        # force the model to run on CPU for compatibility with most environments
        model_kwargs={"device": "cpu"},
        # normalize embeddings to ensure consistent vector magnitudes, which can improve 
        # similarity calculations
        encode_kwargs={"normalize_embeddings": True},
    )

    # Test the model to ensure it's working
    embeddings.embed_query("test")
    model_initialized = True
except Exception as e:
    print(f"Failed to initialize model: {str(e)}")
    model_initialized = False


class TextRequest(BaseModel):
    text: str


class BatchTextRequest(BaseModel):
    texts: List[str]


@app.post("/embed")
async def create_embedding(request: TextRequest):
    if not model_initialized:
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="Model not initialized"
        )
    # embed_query method will return a normalized embedding vector for the input text, 
    # which is suitable for similarity comparisons and other downstream tasks.
    return embeddings.embed_query(request.text)


@app.post("/embed_batch")
async def create_embeddings_batch(request: BatchTextRequest):
    if not model_initialized:
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="Model not initialized"
        )
    # embed_documents will return a list of normalized embedding vectors corresponding 
    # to each input text, which can be used for batch processing and efficient similarity 
    # calculations across multiple texts.
    return embeddings.embed_documents(request.texts)


@app.get("/health")
async def health_check():
    if not model_initialized:
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="Model not initialized"
        )
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
