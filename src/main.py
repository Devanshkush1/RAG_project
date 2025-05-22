from fastapi import FastAPI, UploadFile, File, HTTPException
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import json
from dotenv import load_dotenv
from typing import Dict

load_dotenv()
app = FastAPI()
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
dimension = 384  # Dimension of BAAI/bge-small-en-v1.5 embeddings
index = faiss.IndexFlatL2(dimension)  # FAISS index for L2 distance
chunks_store = []  # Store chunks for retrieval
chunk_metadata = []  # Store metadata for chunks

# Local storage (no MongoDB)
METADATA_FILE = "metadata.json"

def load_metadata():
    if not os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "w") as f:
            json.dump([], f)
    with open(METADATA_FILE, "r") as f:
        return json.load(f)

def save_metadata(metadata):
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

def process_document(file_path: str, filename: str):
    if filename.lower().endswith('.pdf'):
        with open(file_path, "rb") as f:
            pdf = PyPDF2.PdfReader(f)
            text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
            page_count = len(pdf.pages)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            page_count = 1
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=25)
    return splitter.split_text(text), page_count

def store_chunks(chunks, document_id):
    global chunks_store, chunk_metadata
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    index.add(embeddings)
    chunks_store.extend(chunks)
    chunk_metadata.extend([{"document_id": document_id} for _ in chunks])

def retrieve_context(query: str, top_k: int) -> str:
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    return "\n\n".join([chunks_store[idx] for idx in indices[0]])

def call_llm_api(query: str, context: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Gemini API key not found in environment variables")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"Context: {context}\nQuery: {query}"
                    }
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 280
        }
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    response_json = response.json()
    if "candidates" not in response_json:
        error_message = response_json.get("error", {}).get("message", "Unknown error")
        raise HTTPException(status_code=500, detail=f"Gemini API error: {error_message}")
    return response_json["candidates"][0]["content"]["parts"][0]["text"]

@app.post("/upload-documents")
async def upload_documents(files: list[UploadFile] = File(...)):
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 documents allowed")
    metadata = load_metadata()
    for file in files:
        content = await file.read()
        file_path = f"articles/{file.filename}"
        os.makedirs("articles", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(content)
        chunks, page_count = process_document(file_path, file.filename)
        if page_count > 1000:
            raise HTTPException(status_code=400, detail=f"{file.filename} exceeds 1000 pages")
        metadata_record = {
            "filename": file.filename,
            "page_count": page_count,
            "uploaded_at": datetime.utcnow().isoformat()
        }
        metadata.append(metadata_record)
        store_chunks(chunks, file.filename)
    save_metadata(metadata)
    return {"message": "Documents uploaded successfully"}

@app.post("/query")
async def query_documents(request: Dict[str, str]):
    query = request.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")
    context = retrieve_context(query, top_k=3)
    response = call_llm_api(query, context)
    return {"response": response}

@app.get("/documents")
async def get_document_metadata():
    return load_metadata()