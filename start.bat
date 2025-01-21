@echo off
start "Frontend" cmd /k "cd frontend && npm run dev"
start "Backend" cmd /k "cd src && python -m uvicorn backend:app --reload --host 0.0.0.0 --port 8000"