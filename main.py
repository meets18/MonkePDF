import os
import sys
import json
import hashlib
import time
import threading
import requests
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

# Suppress Hugging Face warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="Monke PDF Summarizer & Q&A RAG")

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_DIR = ".pdf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def cleanup_expired_cache():
    """Background task running every minute to delete cache files older than 10 minutes (600 seconds) of inactivity."""
    while True:
        time.sleep(60)
        try:
            now = time.time()
            max_age_seconds = 10 * 60  # 10 minutes
            for filename in os.listdir(CACHE_DIR):
                file_path = os.path.join(CACHE_DIR, filename)
                if os.path.isfile(file_path):
                    mtime = os.path.getmtime(file_path)
                    age = now - mtime
                    if age > max_age_seconds:
                        try:
                            os.remove(file_path)
                            print(f"[*] Background Cleanup: Evicted expired inactive file: {filename}")
                        except Exception as delete_err:
                            print(f"[!] Error deleting expired file {filename}: {delete_err}")
        except Exception as e:
            print(f"[!] Error in background cache cleanup task: {e}")

# Start the daemon background thread
cleanup_thread = threading.Thread(target=cleanup_expired_cache, daemon=True)
cleanup_thread.start()

OLLAMA_DEFAULT_MODEL = "llama3.2"

# Global embedding model variable (lazy-loaded)
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("[*] Loading semantic search model (all-MiniLM-L6-v2)...")
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model

def get_file_hash_bytes(file_bytes):
    """Calculates SHA256 of file bytes."""
    hasher = hashlib.sha256()
    hasher.update(file_bytes)
    return hasher.hexdigest()

def load_from_cache(file_hash):
    """Loads chunks and embeddings from cache. Migrates legacy formats if page_count or char_count are missing."""
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}.json")
    if os.path.exists(cache_path):
        try:
            # Touch both files to prevent them from getting cleaned up due to age during active usage
            os.utime(cache_path, None)
            orig_pdf_path = os.path.join(CACHE_DIR, f"original_{file_hash}.pdf")
            if os.path.exists(orig_pdf_path):
                os.utime(orig_pdf_path, None)
                
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Reconstruct embeddings as numpy arrays
            chunks = data.get("chunks", [])
            for chunk in chunks:
                chunk["embedding"] = np.array(chunk["embedding"], dtype=np.float32)
                
            # Back-fill legacy cache files to prevent KeyError on page_count or char_count
            if "page_count" not in data or "char_count" not in data:
                print(f"[*] Upgrading legacy cache file for {file_hash}...")
                page_count = max([c["page"] for c in chunks]) if chunks else 0
                char_count = sum([len(c["content"]) for c in chunks])
                data["page_count"] = page_count
                data["char_count"] = char_count
                
                # Write back the updated cache format to disk
                try:
                    serializable_chunks = []
                    for chunk in chunks:
                        serializable_chunks.append({
                            "page": chunk["page"],
                            "content": chunk["content"],
                            "embedding": chunk["embedding"].tolist()
                        })
                    upgrade_data = {
                        "file_hash": file_hash,
                        "pdf_path": data.get("pdf_path", ""),
                        "page_count": page_count,
                        "char_count": char_count,
                        "chunks": serializable_chunks
                    }
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(upgrade_data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[!] Warning: Failed to write cache upgrade: {e}")
                    
            return data
        except Exception as e:
            print(f"[!] Warning: Failed to load cache: {e}")
            return None
    return None

def save_to_cache(file_hash, pdf_path, chunks, page_count, char_count):
    """Saves chunks and embeddings to cache."""
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}.json")
    try:
        serializable_chunks = []
        for chunk in chunks:
            serializable_chunks.append({
                "page": chunk["page"],
                "content": chunk["content"],
                "embedding": chunk["embedding"].tolist()
            })
        
        data = {
            "file_hash": file_hash,
            "pdf_path": pdf_path,
            "page_count": page_count,
            "char_count": char_count,
            "chunks": serializable_chunks
        }
        
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[*] Saved cache: {cache_path}")
    except Exception as e:
        print(f"[!] Warning: Failed to save cache: {e}")

def enforce_cache_limit(max_files=5):
    """Purges oldest cache entries (LRU)."""
    try:
        files = [
            os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR)
            if f.endswith(".json")
        ]
        if len(files) <= max_files:
            return
        
        files.sort(key=os.path.getmtime)
        to_delete = len(files) - max_files
        for i in range(to_delete):
            # Also attempt to delete the corresponding original pdf
            cache_file = files[i]
            os.remove(cache_file)
            print(f"[*] Evicted cache file: {cache_file}")
            
            # Extract hash to clean up the cached PDF
            fh = os.path.basename(cache_file).replace(".json", "")
            orig_pdf = os.path.join(CACHE_DIR, f"original_{fh}.pdf")
            if os.path.exists(orig_pdf):
                os.remove(orig_pdf)
                print(f"[*] Evicted PDF file: {orig_pdf}")
    except Exception as e:
        print(f"[!] Warning: Evicting cache failed: {e}")

def run_local_ocr(pdf_path):
    """Fallback OCR processing using PyMuPDF and EasyOCR."""
    try:
        import fitz
        import easyocr
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="OCR dependencies missing on server. Run: pip install easyocr pymupdf"
        )
    
    try:
        print("[*] Backend: Initializing EasyOCR...")
        reader = easyocr.Reader(['en'])
        
        doc = fitz.open(pdf_path)
        pages_text = []
        total_chars = 0
        
        for i, page in enumerate(doc):
            print(f"[*] Backend OCR: processing page {i+1}/{len(doc)}")
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            results = reader.readtext(img_bytes, detail=0)
            page_text = " ".join(results)
            pages_text.append(page_text)
            total_chars += len(page_text)
            
        return pages_text, total_chars
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR parsing failed: {str(e)}")

def parse_pdf_text(pdf_path):
    """Extracts text from PDF. Falls back to OCR if scanned."""
    try:
        reader = PdfReader(pdf_path)
        pages_text = []
        total_chars = 0
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_text.append(text)
                total_chars += len(text)
            else:
                pages_text.append("")
                
        # If no text is found, run OCR
        if total_chars < 100:
            print("[*] Backend: Scanned PDF detected. Invoking OCR...")
            pages_text, total_chars = run_local_ocr(pdf_path)
            
        return pages_text, len(reader.pages), total_chars
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing error: {str(e)}")

def chunk_text(pages_text, chunk_size=600, overlap=100):
    """Splits text pages into overlapping chunks."""
    chunks = []
    for page_num, text in enumerate(pages_text, start=1):
        if not text.strip():
            continue
        words = text.split()
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_content = " ".join(chunk_words)
            chunks.append({
                "page": page_num,
                "content": chunk_content
            })
            if len(chunk_words) < chunk_size:
                break
            i += (chunk_size - overlap)
    return chunks

def retrieve_top_k(query, cached_chunks, k=3):
    """Finds top-K relevant chunks using Cosine Similarity."""
    model = get_embedding_model()
    query_emb = model.encode([query])[0]
    
    similarities = []
    for chunk in cached_chunks:
        dot_product = np.dot(query_emb, chunk["embedding"])
        norm_q = np.linalg.norm(query_emb)
        norm_c = np.linalg.norm(chunk["embedding"])
        sim = dot_product / (norm_q * norm_c) if norm_q > 0 and norm_c > 0 else 0.0
        similarities.append(sim)
        
    top_indices = np.argsort(similarities)[::-1][:k]
    
    results = []
    for idx in top_indices:
        results.append((cached_chunks[idx], float(similarities[idx])))
    return results

# Serve index.html as the landing/base page
@app.get("/", response_class=HTMLResponse)
def read_root():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Error: index.html not found. Make sure index.html is in the project directory.</h3>"

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # 1. Validate file format
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files (.pdf) are allowed.")
        
    # Read bytes to validate file size
    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    print(f"[*] Uploaded PDF size: {file_size_mb:.2f} MB")
    
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds the 50 MB limit.")
        
    # 2. Check cache
    file_hash = get_file_hash_bytes(contents)
    original_pdf_path = os.path.join(CACHE_DIR, f"original_{file_hash}.pdf")
    
    # Save the original PDF for the browser viewer to load
    if not os.path.exists(original_pdf_path):
        with open(original_pdf_path, "wb") as f:
            f.write(contents)
            
    cached_data = load_from_cache(file_hash)
    if cached_data:
        print("[*] Cache hit! Returning cached metadata...")
        return {
            "status": "success",
            "file_hash": file_hash,
            "filename": file.filename,
            "page_count": cached_data["page_count"],
            "char_count": cached_data["char_count"],
            "cached": True
        }
        
    # 3. Process PDF
    print("[*] Cache miss! Extracting PDF contents...")
    pages_text, page_count, char_count = parse_pdf_text(original_pdf_path)
    chunks = chunk_text(pages_text)
    
    if not chunks:
        # Clean up the PDF if it contains no text and OCR returned nothing
        if os.path.exists(original_pdf_path):
            os.remove(original_pdf_path)
        raise HTTPException(status_code=400, detail="Could not extract readable text from this PDF.")
        
    # 4. Generate Embeddings
    print("[*] Generating embeddings for RAG database...")
    model = get_embedding_model()
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts)
    
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
        
    # 5. Save to Cache and Enforce Limit
    save_to_cache(file_hash, original_pdf_path, chunks, page_count, char_count)
    enforce_cache_limit(max_files=5)
    
    return {
        "status": "success",
        "file_hash": file_hash,
        "filename": file.filename,
        "page_count": page_count,
        "char_count": char_count,
        "cached": False
    }

@app.get("/pdf/{file_hash}")
def get_pdf(file_hash: str):
    """Serves the original PDF file directly to browser iframe."""
    pdf_path = os.path.join(CACHE_DIR, f"original_{file_hash}.pdf")
    if os.path.exists(pdf_path):
        try:
            # Touch both files to prevent cleanup during active viewing
            os.utime(pdf_path, None)
            json_path = os.path.join(CACHE_DIR, f"{file_hash}.json")
            if os.path.exists(json_path):
                os.utime(json_path, None)
        except Exception as e:
            print(f"[!] Warning: Failed to touch served files: {e}")
        return FileResponse(pdf_path, media_type="application/pdf")
    raise HTTPException(status_code=404, detail="PDF not found.")

@app.post("/summarize")
def summarize_endpoint(payload: dict):
    """Generates document summary using the first few chunks."""
    file_hash = payload.get("file_hash")
    if not file_hash:
        raise HTTPException(status_code=400, detail="Missing file_hash")
        
    cached_data = load_from_cache(file_hash)
    if not cached_data:
        raise HTTPException(status_code=404, detail="Document data not found or evicted from cache")
        
    chunks = cached_data["chunks"]
    summary_chunks = chunks[:4]
    summary_context = "\n\n".join([f"--- Section {i+1} ---\n{c['content']}" for i, c in enumerate(summary_chunks)])
    
    summary_prompt = (
        "You are an expert document summarizer.\n"
        "Provide a concise, high-level executive summary of the following document sections.\n"
        "Highlight the main topics, key objectives, and main conclusions.\n\n"
        f"Document Content:\n{summary_context}\n\n"
        "Summary:"
    )
    
    # We query Ollama and stream the summary chunk by chunk
    def generate_summary():
        url = "http://localhost:11434/api/chat"
        payload_data = {
            "model": OLLAMA_DEFAULT_MODEL,
            "messages": [{"role": "user", "content": summary_prompt}],
            "stream": True
        }
        try:
            response = requests.post(url, json=payload_data, stream=True)
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    text = data.get("message", {}).get("content", "")
                    if text:
                        yield text
        except Exception as e:
            yield f"[Ollama Connection Error: {e}. Ensure Ollama is running.]"
            
    return StreamingResponse(generate_summary(), media_type="text/plain")

@app.post("/query")
def query_endpoint(payload: dict):
    """Streams Q&A chat responses with dynamic RAG context and chat memory."""
    file_hash = payload.get("file_hash")
    query = payload.get("query")
    doc_mode = payload.get("doc_mode", "strict")
    chat_history = payload.get("chat_history", [])
    
    if not file_hash or not query:
        raise HTTPException(status_code=400, detail="Missing file_hash or query")
        
    cached_data = load_from_cache(file_hash)
    if not cached_data:
        raise HTTPException(status_code=404, detail="Document data not found or evicted from cache")
        
    chunks = cached_data["chunks"]
    
    # Find relevant chunks using Cosine Similarity
    retrieved_results = retrieve_top_k(query, chunks, k=3)
    
    # Format retrieved passages
    context_parts = []
    for chunk, score in retrieved_results:
        context_parts.append(f"[Source: Page {chunk['page']}, Relevance: {score:.2f}]\n{chunk['content']}")
    context_str = "\n\n".join(context_parts)
    
    # Build System Prompt based on Mode
    if doc_mode == "strict":
        system_content = (
            "You are a precise question-answering assistant.\n"
            "Use the provided context from the PDF to answer the question.\n"
            "If the context does not contain the answer, say 'I cannot find the answer in the document.'\n"
            "Do not make up facts or extrapolate outside the context."
        )
    else:
        system_content = (
            "You are a helpful assistant.\n"
            "Answer the user's question. You may use the provided PDF context if it is relevant to help answer the question.\n"
            "If the context is insufficient or does not mention the topic, feel free to draw on your general knowledge to fully answer or explain the user's question, but clarify that it is general knowledge."
        )
        
    # Construct complete message thread
    messages = [
        {"role": "system", "content": system_content}
    ]
    
    # Append conversation memory
    messages.extend(chat_history)
    
    # Append latest query with context
    messages.append({
        "role": "user",
        "content": f"Context:\n{context_str}\n\nQuestion: {query}"
    })
    
    def generate_chat():
        url = "http://localhost:11434/api/chat"
        payload_data = {
            "model": OLLAMA_DEFAULT_MODEL,
            "messages": messages,
            "stream": True
        }
        try:
            response = requests.post(url, json=payload_data, stream=True)
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    text = data.get("message", {}).get("content", "")
                    if text:
                        yield text
        except Exception as e:
            yield f"[Ollama Connection Error: {e}. Ensure Ollama is running.]"
            
    return StreamingResponse(generate_chat(), media_type="text/plain")

# Mount static files (HTML, CSS, JS) if placed in static directories
# Here, we serve index.html directly, and serve app.js and style.css as basic static files.
@app.get("/app.js")
def get_js():
    js_path = os.path.join(os.path.dirname(__file__), "app.js")
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="app.js not found")

@app.get("/style.css")
def get_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css not found")

@app.get("/placeholder.png")
def get_placeholder():
    path = os.path.join(os.path.dirname(__file__), "placeholder.png")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    raise HTTPException(status_code=404, detail="placeholder.png not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
