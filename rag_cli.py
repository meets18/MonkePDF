import os
import sys
import argparse
import json
import requests
import numpy as np
from pypdf import PdfReader

# Suppress Hugging Face warnings and logs
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings
warnings.filterwarnings("ignore")

# Define default model names
OLLAMA_DEFAULT_MODEL = "llama3.2"
HF_DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

def run_local_ocr(pdf_path):
    """Fallback OCR processing for scanned PDFs using PyMuPDF and EasyOCR."""
    try:
        import fitz  # PyMuPDF
        import easyocr
    except ImportError:
        print("\n[!] Error: OCR dependencies are missing.")
        print("[!] Please install them by running: pip install easyocr pymupdf")
        print("[!] Note: EasyOCR will download its English detection model (~100MB) automatically on the first run.")
        sys.exit(1)
        
    try:
        print("[*] Initializing EasyOCR reader (English)...")
        # EasyOCR auto-detects GPU/CUDA if available
        reader = easyocr.Reader(['en'])
        
        doc = fitz.open(pdf_path)
        pages_text = []
        total_chars = 0
        
        print(f"[*] Processing {len(doc)} pages with local OCR (scanned PDF)...")
        for i, page in enumerate(doc):
            print(f"  [-] OCR processing page {i+1}/{len(doc)}... ", end="", flush=True)
            # Render page to a pixmap image (150 DPI is a good speed/accuracy balance)
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            
            # Extract text from the image bytes
            results = reader.readtext(img_bytes, detail=0)
            page_text = " ".join(results)
            pages_text.append(page_text)
            total_chars += len(page_text)
            print(f"Done ({len(page_text)} chars)")
            
        print(f"[*] Successfully completed OCR for {len(doc)} pages (Total characters: {total_chars:,})")
        return pages_text
    except Exception as e:
        print(f"\n[!] Error during local OCR: {e}")
        sys.exit(1)

def load_pdf(pdf_path):
    """Extracts text from the PDF file and returns it as a list of page strings. Falls back to OCR if scanned."""
    print(f"[*] Reading PDF: {pdf_path}...")
    if not os.path.exists(pdf_path):
        print(f"[!] Error: File not found at {pdf_path}")
        sys.exit(1)
    
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
                
        # If total extracted character count is extremely low, it's likely a scanned PDF
        if total_chars < 100:
            print("[!] Scanned PDF detected (no selectable text found). Initializing local OCR fallback...")
            pages_text = run_local_ocr(pdf_path)
        else:
            print(f"[*] Successfully parsed {len(reader.pages)} pages (Total characters: {total_chars:,})")
            
        return pages_text
    except Exception as e:
        print(f"[!] Error parsing PDF: {e}")
        sys.exit(1)

def chunk_text(pages_text, chunk_size=600, overlap=100):
    """Splits text into chunks of chunk_size with overlap, keeping page references."""
    print("[*] Chunking text and preparing context...")
    chunks = []
    
    for page_num, text in enumerate(pages_text, start=1):
        if not text.strip():
            continue
        
        words = text.split()
        # Move through the words in steps
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
            
    print(f"[*] Created {len(chunks)} chunks from the document.")
    return chunks

def embed_chunks(chunks):
    """Computes embeddings for each text chunk using sentence-transformers."""
    print("[*] Initializing embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"[!] Error loading embedding model: {e}")
        sys.exit(1)
        
    print("[*] Generating embeddings for document chunks...")
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
        
    return model, chunks

def retrieve_top_k(query, embedding_model, chunks, k=3):
    """Retrieves top k chunks closest to the query embedding using cosine similarity."""
    query_emb = embedding_model.encode([query])[0]
    
    similarities = []
    for chunk in chunks:
        # Cosine similarity
        dot_product = np.dot(query_emb, chunk["embedding"])
        norm_q = np.linalg.norm(query_emb)
        norm_c = np.linalg.norm(chunk["embedding"])
        sim = dot_product / (norm_q * norm_c) if norm_q > 0 and norm_c > 0 else 0.0
        similarities.append(sim)
        
    top_indices = np.argsort(similarities)[::-1][:k]
    
    results = []
    for idx in top_indices:
        results.append((chunks[idx], similarities[idx]))
    return results

def query_ollama(messages, model=OLLAMA_DEFAULT_MODEL, stream=True):
    """Sends a query to local Ollama API using the chat endpoint."""
    url = "http://localhost:11434/api/chat"
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
        
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream
    }
    
    try:
        response = requests.post(url, json=payload, stream=stream)
        if response.status_code != 200:
            print(f"\n[!] Ollama error: Status code {response.status_code}")
            return None
            
        if stream:
            full_response = []
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    message = data.get("message", {})
                    text = message.get("content", "")
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    full_response.append(text)
                    if data.get("done", False):
                        break
            print() # Print final newline
            return "".join(full_response)
        else:
            data = response.json()
            return data.get("message", {}).get("content", "")
    except Exception as e:
        print(f"\n[!] Connection to Ollama failed: {e}")
        print("[!] Make sure Ollama is running and has the model pulled.")
        return None

def query_huggingface(messages, hf_pipeline, tokenizer, stream=True):
    """Generates text using a local Hugging Face model with a conversation history."""
    from transformers import TextStreamer
    
    if isinstance(messages, str):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided PDF context."},
            {"role": "user", "content": messages}
        ]
    
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(hf_pipeline.device)
        
        if stream:
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            # Standard Hugging Face pipelines or generate method
            # We want to capture the generated tokens to return them.
            # To do that while streaming, we can use a custom streamer or just let it generate.
            # Since returning the full text is needed for the chat history, let's use generation
            # and print using the TextStreamer.
            generated_ids = hf_pipeline.generate(
                **model_inputs,
                max_new_tokens=512,
                streamer=streamer
            )
            # Decode the generated response
            input_length = model_inputs.input_ids.shape[1]
            generated_text = tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True)
            return generated_text
        else:
            generated_ids = hf_pipeline.generate(
                **model_inputs,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
    except Exception as e:
        print(f"\n[!] HuggingFace generation failed: {e}")
        return None

def get_file_hash(file_path):
    """Calculates the SHA256 hash of a file to uniquely identify it."""
    import hashlib
    hasher = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"[!] Error hashing file {file_path}: {e}")
        sys.exit(1)

def load_from_cache(cache_dir, file_hash):
    """Loads chunks and embeddings from the cache if they exist."""
    cache_path = os.path.join(cache_dir, f"{file_hash}.json")
    if os.path.exists(cache_path):
        try:
            # Touch the file to update modification time (for LRU eviction)
            os.utime(cache_path, None)
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Convert embedding lists back to numpy arrays
            for chunk in data["chunks"]:
                chunk["embedding"] = np.array(chunk["embedding"], dtype=np.float32)
            return data
        except Exception as e:
            print(f"[!] Warning: Failed to load cache from {cache_path}: {e}")
            return None
    return None

def save_to_cache(cache_dir, file_hash, pdf_path, chunks):
    """Saves chunks and embeddings to the cache."""
    cache_path = os.path.join(cache_dir, f"{file_hash}.json")
    try:
        # Create a serializable copy of chunks
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
            "chunks": serializable_chunks
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[*] Saved embeddings to cache: {cache_path}")
    except Exception as e:
        print(f"[!] Warning: Failed to write cache: {e}")

def enforce_cache_limit(cache_dir, max_files=5):
    """Deletes oldest cache files if the number of files exceeds max_files (LRU)."""
    try:
        files = [
            os.path.join(cache_dir, f) for f in os.listdir(cache_dir)
            if f.endswith(".json")
        ]
        if len(files) <= max_files:
            return
            
        # Sort files by modification time (oldest first)
        files.sort(key=os.path.getmtime)
        
        # Delete oldest files until we are at the limit
        to_delete = len(files) - max_files
        for i in range(to_delete):
            os.remove(files[i])
            print(f"[*] Evicted oldest cache file from store: {files[i]}")
    except Exception as e:
        print(f"[!] Warning: Failed to enforce cache limit: {e}")

def main():
    parser = argparse.ArgumentParser(description="Local PDF RAG Summarizer & Q&A CLI")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file")
    parser.add_argument("--mode", type=str, choices=["ollama", "huggingface"], default="ollama",
                        help="LLM execution mode (ollama or huggingface)")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model name to use (default: llama3.2 for ollama, Qwen/Qwen2.5-0.5B-Instruct for hf)")
    parser.add_argument("--doc-mode", type=str, choices=["strict", "open"], default="strict",
                        help="Answering mode: 'strict' (restrict to PDF content only) or 'open' (allow general knowledge explanation)")
    
    args = parser.parse_args()
    
    # 0. Validate file existence and check size limit (50MB)
    if not os.path.exists(args.pdf):
        print(f"[!] Error: File not found at {args.pdf}")
        sys.exit(1)
        
    file_size_bytes = os.path.getsize(args.pdf)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"[*] PDF file size: {file_size_mb:.2f} MB")
    
    if file_size_bytes > 50 * 1024 * 1024:
        print(f"[!] Error: File size exceeds the 50 MB limit. Please use a smaller PDF.")
        sys.exit(1)
    
    # 1. Setup execution mode
    mode = args.mode
    model_name = args.model
    doc_mode = args.doc_mode
    
    hf_model = None
    hf_tokenizer = None
    
    if mode == "ollama":
        if not model_name:
            model_name = OLLAMA_DEFAULT_MODEL
        print(f"[*] Mode: Ollama. Using model '{model_name}'")
        # Quick health check to see if Ollama is running
        try:
            requests.get("http://localhost:11434/", timeout=2)
        except requests.exceptions.RequestException:
            print("[!] Warning: Could not connect to local Ollama. Is it running?")
            print("[!] Please run Ollama App or ensure the service is started.")
            sys.exit(1)
            
    elif mode == "huggingface":
        if not model_name:
            model_name = HF_DEFAULT_MODEL
        print(f"[*] Mode: Hugging Face. Using model '{model_name}'")
        print("[*] Loading Hugging Face tokenizer and model (this might download ~950MB on first run)...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Running on device: {device}")
        
        try:
            hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
        except Exception as e:
            print(f"[!] Error loading Hugging Face model: {e}")
            sys.exit(1)
            
    # 2. Check Embedding Cache
    file_hash = get_file_hash(args.pdf)
    cache_dir = ".pdf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    cached_data = load_from_cache(cache_dir, file_hash)
    embedding_model = None
    
    if cached_data:
        print("[*] Found cached embeddings! Skipping text extraction and embedding generation...")
        chunks = cached_data["chunks"]
    else:
        # Extract and Chunk PDF
        pages_text = load_pdf(args.pdf)
        chunks = chunk_text(pages_text)
        
        if not chunks:
            print("[!] Error: No readable text found in PDF.")
            sys.exit(1)
            
        # Generate Embeddings
        embedding_model, chunks = embed_chunks(chunks)
        
        # Save to cache
        save_to_cache(cache_dir, file_hash, args.pdf, chunks)
        enforce_cache_limit(cache_dir, max_files=5)
    
    # 4. Generate Document Summary
    print("\n" + "="*50)
    print("      GENERATING DOCUMENT SUMMARY")
    print("="*50)
    
    # Compile a representation of the PDF for summarization
    # We will take the first few chunks (up to 3 chunks or ~1500 words)
    summary_chunks = chunks[:4]
    summary_context = "\n\n".join([f"--- Section {i+1} ---\n{c['content']}" for i, c in enumerate(summary_chunks)])
    
    summary_prompt = (
        "You are an expert document summarizer.\n"
        "Provide a concise, high-level executive summary of the following document sections.\n"
        "Highlight the main topics, key objectives, and main conclusions.\n\n"
        f"Document Content:\n{summary_context}\n\n"
        "Summary:"
    )
    
    print("[*] Model output:\n")
    if mode == "ollama":
        query_ollama(summary_prompt, model=model_name, stream=True)
    else:
        query_huggingface(summary_prompt, hf_model, hf_tokenizer, stream=True)
        
    print("="*50 + "\n")
    
    # 5. Q&A Loop
    print("Entering Interactive Q&A Mode. Type 'exit' or 'quit' to stop.")
    print(f"[*] Answering Mode: {doc_mode.upper()} (strict = PDF-only, open = general knowledge allowed)")
    print("[*] Tip: Type '/mode' during chat to toggle between strict and open modes.")
    chat_history = []
    
    while True:
        try:
            query = input("\nAsk a question about the PDF > ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
            
        if query.strip().lower() in ["exit", "quit"]:
            print("Exiting...")
            break
            
        if not query.strip():
            continue
            
        # Parse mode command toggles
        if query.strip().lower() == "/mode":
            doc_mode = "open" if doc_mode == "strict" else "strict"
            print(f"[*] Mode toggled. Answering Mode is now: {doc_mode.upper()}")
            continue
        elif query.strip().lower().startswith("/mode "):
            val = query.strip().split(None, 1)[1].lower()
            if val in ["strict", "open"]:
                doc_mode = val
                print(f"[*] Answering Mode set to: {doc_mode.upper()}")
            else:
                print("[!] Invalid mode. Choose 'strict' or 'open'.")
            continue
            
        if embedding_model is None:
            print("[*] Loading semantic search index (this runs once)...")
            from sentence_transformers import SentenceTransformer
            try:
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print(f"[!] Error loading embedding model: {e}")
                sys.exit(1)
                
        print("[*] Searching PDF database for relevant content...")
        retrieved = retrieve_top_k(query, embedding_model, chunks, k=3)
        
        # Format Context
        context_parts = []
        for i, (chunk, score) in enumerate(retrieved, start=1):
            context_parts.append(f"[Source: Page {chunk['page']}, Relevance: {score:.2f}]\n{chunk['content']}")
            
        context_str = "\n\n".join(context_parts)
        
        # Set prompt system contents depending on active mode
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
            
        # Build prompt list with chat history (memory)
        messages = [
            {
                "role": "system",
                "content": system_content
            }
        ]
        
        # Add past chat history
        messages.extend(chat_history)
        
        # Add the current query along with the retrieved context
        messages.append({
            "role": "user",
            "content": f"Context:\n{context_str}\n\nQuestion: {query}"
        })
        
        print(f"\n[*] Top 3 Relevant Passages Found:")
        for idx, (chunk, score) in enumerate(retrieved, start=1):
            print(f"  {idx}. Page {chunk['page']} (score: {score:.2f}): {chunk['content'][:80]}...")
            
        print(f"\n[*] Answer from LLM (Mode: {doc_mode.upper()}):")
        answer = None
        if mode == "ollama":
            answer = query_ollama(messages, model=model_name, stream=True)
        else:
            answer = query_huggingface(messages, hf_model, hf_tokenizer, stream=True)
            print()
            
        # If successfully generated, append to chat history (without the heavy context block)
        if answer:
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
