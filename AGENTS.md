# AGENTS.md

## Cursor Cloud specific instructions

This is a VFX Motion Capture monorepo with a Python FastAPI backend (`backend/`) and a Next.js frontend (`frontend/`).

### Services

| Service | Command | Port |
|---------|---------|------|
| Backend (FastAPI) | `cd /workspace && uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000` | 8000 |
| Frontend (Next.js) | `cd /workspace/frontend && npm run dev` | 3000 |
| Redis | `redis-server --daemonize yes` | 6379 |
| Celery Worker | `cd /workspace && celery -A backend.workers.celery_app worker --loglevel=info` | — |

Redis must be started before the backend. The backend must be started before the frontend (frontend proxies API calls via `next.config.js` rewrites to `localhost:8000`).

### Gotchas

- **`.env` CORS_ORIGINS format**: The `.env.example` has `CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000` but pydantic-settings requires JSON list format for `List[str]` fields. The `.env` must use `CORS_ORIGINS=["http://localhost:3000","http://127.0.0.1:3000"]`.
- **`python3-dev` required**: The `insightface` package requires `python3-dev` for compilation headers (`Python.h`). This is a system dependency pre-installed by the update script.
- **No test files exist yet**: `pytest` exits with code 5 (no tests collected). The test infrastructure (pytest, pytest-asyncio) is installed.
- **`ruff` not in requirements.txt**: The linter must be installed separately (`pip install ruff`). The `Makefile` `lint` target expects it.
- **Frontend `.eslintrc.json`**: Not committed to the repo. `next lint` will prompt interactively without it. The update script creates it automatically.
- **No GPU required**: The app starts and serves the UI/API without GPU. AI model inference requires GPU + downloaded model weights, but all non-inference functionality works on CPU.
- **Lint/test commands**: See `Makefile` for `make lint`, `make test`, `make test-backend`. Backend lint: `ruff check backend/`. Frontend lint: `cd frontend && npm run lint`.
