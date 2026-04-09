# Agent Instructions — ProjectX Backend

## Project Overview

ProjectX is a multi-tenant SaaS platform built with FastAPI (Python 3.12) and PostgreSQL. It exposes a REST API consumed by a React frontend and third-party integrations. The codebase lives in the `backend/` directory. Do not modify `frontend/` unless explicitly asked.

## Core Principles

- **Correctness first.** Never sacrifice correctness for brevity. If a change is risky, say so.
- **Minimal surface area.** Prefer editing existing files over creating new ones.
- **No silent failures.** Every error path must either raise an exception or log at WARNING or above.
- **Type everything.** All Python functions must have full type annotations. Avoid `Any` unless wrapping a third-party interface.

## Repository Layout

```
backend/
  app/
    main.py          # FastAPI app factory
    config.py        # Settings via pydantic-settings
    models/          # SQLAlchemy ORM models
    schemas/         # Pydantic request/response schemas
    routers/         # API route handlers
    services/        # Business logic layer
    repositories/    # Database access layer
    middleware/       # Auth, logging, error handling
  migrations/        # Alembic migration files
  tests/
  requirements.txt
```

## Code Style

- Follow PEP 8. Line length 100.
- Use `ruff` for linting (`ruff check .`) and `black` for formatting (`black .`).
- Imports: stdlib → third-party → local, separated by blank lines.
- Prefer `pathlib.Path` over `os.path`.
- Use `logging.getLogger(__name__)` per module; never use `print` in production code.

## Running the App

```bash
# Development
uvicorn app.main:app --reload --port 8000

# With docker-compose
docker-compose up --build
```

## Environment Variables

All configuration lives in `backend/.env` (never committed). See `app/config.py` for the full schema. Required:
- `DATABASE_URL` — PostgreSQL connection string
- `SECRET_KEY` — 64-byte hex string for JWT signing
- `REDIS_URL` — for caching and Celery broker
- `ENVIRONMENT` — `development` | `staging` | `production`

## Commit Conventions

Use Conventional Commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`. Keep subject lines under 72 characters. Reference issue numbers when applicable (`closes #123`).
