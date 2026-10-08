# Local Knowledge

A local-first **Chat with Documents** workshop app. Upload PDF and DOCX course material, extract text, split it into overlapping chunks, create local embeddings with Ollama, rank those chunks with cosine similarity, and ask a local chat model to answer from the retrieved excerpts.

## Requirements

- Windows 10/11, Python 3.10+, and a current Ollama installation.
- Enough memory for the selected local models. The default chat model is a small 3B model; larger models need more memory.
- Scanned/image-only PDFs are not OCR'd. Use a text-searchable PDF or convert it to text first.

## Run on Windows

1. Install Ollama. With Windows Package Manager, run `winget install --id Ollama.Ollama` in PowerShell, then open a new terminal if the `ollama` command is not available.
2. Download the default local models:

   ```powershell
   ollama run qwen2.5:0.5b
   ollama pull nomic-embed-text
   ```

   Ollama normally runs its local service after installation. If needed, start it with `ollama serve` in a separate terminal.
3. In the project root, create and activate a virtual environment and install Python packages:

   ```powershell
   py -3.10 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Optional: copy `.env.example` to `.env` and adjust the model names or settings. The defaults work with the two commands above.
5. Start the application:

   ```powershell
   uvicorn app.main:app --reload
   ```

6. Open `http://127.0.0.1:8000` in a browser. Upload a PDF or DOCX, wait for indexing to finish, then ask a question. The status in the header reports whether both configured Ollama models are available.

Run the focused tests with `pytest`.

## How the RAG flow works

1. **Extract:** `pypdf` reads PDF page text and page numbers. `python-docx` reads DOCX paragraphs and table rows.
2. **Chunk:** the extracted text is split on words into 180-word chunks with 30 words of overlap. Overlap helps retain a thought that crosses a chunk boundary. Tune `CHUNK_SIZE` and `CHUNK_OVERLAP` in `.env`; overlap must be smaller than chunk size.
3. **Embed:** Ollama's `/api/embed` endpoint creates a vector for every chunk using `nomic-embed-text`. The same model embeds the question.
4. **Retrieve:** NumPy computes cosine similarity between the question vector and each stored chunk vector. The top four matches (`TOP_K`) are selected. Cosine similarity compares direction, not raw vector magnitude; a larger score means a closer semantic direction, not a probability.
5. **Generate:** the selected excerpts and the question are sent to Ollama's `/api/chat` endpoint with instructions to answer only from the evidence. The UI shows the retrieved text, source filename/page, and similarity score so students can inspect the grounding.

## Workshop architecture

- `app/rag.py` contains document parsing, word chunking, metadata, and cosine similarity as small testable functions.
- `app/main.py` provides the FastAPI endpoints and calls the local Ollama HTTP API.
- `app/static/` is a dependency-free browser interface served by FastAPI.
- `tests/test_rag.py` checks chunking, cosine similarity, source metadata, filename handling, and DOCX extraction.

Uploaded document content, chunks, and embeddings are held **in process memory only**. They disappear when the application restarts; this workshop version has no persistent database or multi-user isolation. Keep the service on your local machine and upload only documents you are allowed to use.

## Troubleshooting

- **Ollama is offline:** check that Ollama is running and that `OLLAMA_BASE_URL` points to its local API, normally `http://127.0.0.1:11434`.
- **Model missing:** use `ollama list` and pull the exact names configured by `OLLAMA_CHAT_MODEL` and `OLLAMA_EMBED_MODEL`.
- **Upload says no readable text:** scanned PDFs contain page images rather than selectable text; OCR is outside this app's scope.
- **Answers are slow:** the first request may load a model into memory. Try a smaller chat model or reduce the number/size of retrieved chunks.
- **No answer from documents:** ask a more specific question, check the displayed retrieved evidence, or add the page containing the information. Retrieval is similarity-based and can miss relevant wording.