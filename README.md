# 🐒 Monke PDF - Local PDF Summarizer & Q&A (RAG)

**Monke PDF** is a premium, privacy-first document intelligence platform that allows you to upload, view, summarize, and chat with PDF documents completely offline. Powered by a **FastAPI** backend and a stunning glassmorphic SPA frontend, it runs entirely on your local machine using local embedding models and offline LLMs (via Ollama or Hugging Face). No API keys, no monthly fees, and 100% data confidentiality.

---

## 🎨 Interface Features

- **Automated Splash Page**: A sleek 5-second intro screen presenting a custom 3D glassmorphic brand graphic, character-by-character typewriter animation, and loading indicator.
- **Triple-Box Workspace Dashboard**:
  1. **Document Viewer**: An embedded original PDF viewer utilizing a native iframe with scroll/view activity heartbeats.
  2. **Document Insights**: Displays page count, character metrics, file size, an answering mode toggle, and a dynamically streamed executive summary.
  3. **Interactive Q&A Chat**: A multi-turn chatbot scroll pane supporting conversational history memory and token-by-token text streams parsed with rich Markdown rendering.
- **Dual Answering Modes**:
  - **STRICT Mode**: Directs the LLM to restrict answers *solely* to the contents of the PDF context. If it isn't in the document, the model will politely decline, avoiding AI hallucinations.
  - **OPEN Mode**: Instructs the LLM to use the PDF context as a baseline but allows it to draw on background knowledge to explain definitions or provide external follow-up answers.
- **Close & Reset Interceptors**: Provides confirmation popup modal alerts before closing the loaded PDF or starting a new upload session, protecting you from losing active Q&A history.

---

## 🛠️ Local RAG Architecture

1. **Text Extraction**: Uses local `pypdf` parsing.
2. **Auto-OCR Fallback**: If a document is scanned or contains non-selectable text (less than 100 characters total), the backend automatically processes page views via local `fitz` (PyMuPDF) and runs deep learning character recognition using `easyocr`.
3. **Semantic Indexing**: Text pages are split into overlapping segments and mapped into 384-dimensional vector embeddings using the local `sentence-transformers/all-MiniLM-L6-v2` model.
4. **Vector Retrieve-and-Rank**: Computes cosine similarities between query embeddings and cached segments using NumPy arrays to retrieve the top 3 relevant passages.
5. **Inactivity Auto-Cleanup Daemon**: Cache databases and temporary PDFs are swept every 60 seconds. Files are purged after **10 minutes of inactivity** (resets automatically via document view, heartbeat, or Q&A query events).

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.10+**
- **Ollama** installed on your system (Download from [ollama.com](https://ollama.com)).
- Pull the default local LLM:
  ```bash
  ollama pull llama3.2
  ```

### Installation
1. **Clone or copy this repository** to your project directory.
2. **Set up the virtual environment**:
   ```bash
   python -m venv myenv
   ```
3. **Activate the virtual environment**:
   - **Windows (Command Prompt)**:
     ```cmd
     myenv\Scripts\activate
     ```
   - **Windows (PowerShell)**:
     ```powershell
     .\myenv\Scripts\Activate.ps1
     ```
   - **macOS / Linux**:
     ```bash
     source myenv/bin/activate
     ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: On the first run, the SentenceTransformer and EasyOCR models will download their weights automatically ~150MB total).*

### Running the Web Application
1. Start the backend server:
   ```bash
   python main.py
   ```
2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```
3. Enjoy the splash intro and start uploading PDFs!

---

## 💻 Command-Line Interface (CLI)

You can also run the RAG pipeline directly from your terminal using `rag_cli.py`:

```bash
# Run using the default Ollama model in strict mode
python rag_cli.py --pdf "path/to/your/document.pdf"

# Run in open mode (allowing general knowledge expansion)
python rag_cli.py --pdf "path/to/your/document.pdf" --doc-mode open

# Run using Hugging Face pipelines locally (requires PyTorch GPU/CPU)
python rag_cli.py --pdf "path/to/your/document.pdf" --mode huggingface
```

---

## 📂 Project Directory Structure

```
├── .pdf_cache/           # Temporary local storage for index databases and backup PDFs (Git ignored)
├── explanations/          # Line-by-line documentation text files (Git ignored)
├── myenv/                # Local Python virtual environment files (Git ignored)
├── app.js                # Frontend SPA controller and endpoint stream interface
├── index.html            # Core SPA HTML5 markup structures and modals layout
├── main.py               # FastAPI backend web app, background cleanup daemons, and endpoints
├── placeholder.png       # Custom landing page illustration asset
├── rag_cli.py            # Local RAG CLI terminal tool
├── requirements.txt      # Project Python dependencies list
├── style.css             # Main stylesheet implementing glassmorphic rules and keyframe animations
└── README.md             # This document
```

---

## 🔒 Privacy & Data Confidentiality

Monke PDF is built from the ground up for strict security. Your documents, extracted text, embeddings, queries, and summaries are stored entirely in your project folder (`.pdf_cache`) and generated locally. Absolutely no network calls are made to external SaaS providers, ensuring complete privacy for contract reviews, financials, and legal archives.
