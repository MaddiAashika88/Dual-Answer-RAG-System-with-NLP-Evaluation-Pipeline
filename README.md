# RAG Dual-Answer AI

## Setup

### 1. Get a free Groq API key
- Go to https://console.groq.com
- Sign up for free (no credit card required)
- Create an API key
- Paste it into `backend/main.py` where it says `YOUR_GROQ_API_KEY_HERE`

### 2. Backend (Python)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Backend runs at: http://localhost:8000

### 3. Frontend
Open `frontend/index.html` directly in your browser,
OR serve it with any static server:

```bash
cd frontend
python -m http.server 3000
```

Then open: http://localhost:3000

## How it Works

- **RAG Answer**: Retrieves the top-4 most relevant post-2023 documents from ChromaDB, then feeds them as context to Llama 3.3 70B via Groq.
- **Memory Answer**: Asks the same model the same question but with NO injected context — just its trained knowledge (capped at early 2023).

## Adding Your Own Documents

```bash
curl -X POST http://localhost:8000/add_document \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document text here", "source": "Your Source Name"}'
```
