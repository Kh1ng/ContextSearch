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
python demo/run_demo.py
```

macOS / Linux:
```bash
source .venv/bin/activate
python demo/run_demo.py
```

The demo now prompts you for your query at runtime.
It automatically uses Ollama with `qwen2.5` for generation.

---

## Example Prompts to Paste

```bash
How do I set up JWT authentication?
How do I deploy to production?
What do I need to do to migrate database schema and deploy my database on a friday?
```

Output is saved automatically to `demo/output/`.
