# ContextSearch — Setup & Run Guide

## Prerequisites (install once)

1. Install [Ollama](https://ollama.com) for your platform
2. Install Python 3.12+

---

## Setup (install once)

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

ollama pull nomic-embed-text
ollama pull qwen2.5
```

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull nomic-embed-text
ollama pull qwen2.5
```

---

## Running the Demo

**Terminal 1 — keep open:**
```powershell
ollama serve
```

**Terminal 2 — run the demo:**

Windows:
```powershell
.venv\Scripts\Activate.ps1
python demo/run_demo.py --llm --model qwen2.5 --query "What do I need to do to migrate database schema and deploy my database on a friday?"
```

macOS / Linux:
```bash
source .venv/bin/activate
python demo/run_demo.py --llm --model qwen2.5 --query "What do I need to do to migrate database schema and deploy my database on a friday?"
```

---

## Other Example Queries

```bash
python demo/run_demo.py --llm --model qwen2.5 --query "How do I set up JWT authentication?"
python demo/run_demo.py --llm --model qwen2.5 --query "How do I deploy to production?"
python demo/run_demo.py --llm --model qwen2.5   # runs all 5 default queries
```

Output is saved automatically to `demo/output/`.
