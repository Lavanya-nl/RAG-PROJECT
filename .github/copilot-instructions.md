# Local Knowledge Workspace Instructions

This workspace contains a local-first RAG workshop app built with FastAPI, Ollama, NumPy, and a dependency-free browser interface. Keep the implementation beginner-readable and preserve local-only document/model processing.

## Project Checklist

- [x] Create this workspace instruction file.
- [x] Confirm project requirements: local PDF/DOCX chat using FastAPI and Ollama.
- [x] Scaffold and implement the application.
- [x] Customize the document, retrieval, API, and browser workflows.
- [x] Check extension requirements; no additional extension is required for this project.
- [x] Install dependencies and run tests: `python -m pytest tests -q`.
- [x] Complete `README.md` with Windows setup and workshop architecture.
- [x] Start the application after the user confirmed launch.
- [x] Create and run the VS Code task after launch confirmation.

## Implementation Notes

- `app/rag.py` owns testable extraction, chunking, source metadata, and cosine similarity functions.
- `app/main.py` owns the FastAPI routes and local Ollama API calls.
- `app/static/` contains the frontend; do not add remote JS or CSS dependencies.
- Uploaded content, vectors, and metadata are process-local and reset on restart.
- Keep source filename/page metadata visible in retrieval responses and ground model answers in retrieved excerpts.
- Update focused tests when changing parsing, chunking, similarity, API contracts, or source handling.
