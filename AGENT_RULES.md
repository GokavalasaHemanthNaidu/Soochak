# AI Agent Rules & Guidelines

Welcome to the SOOCHAK workspace! All AI models/agents MUST adhere to these rules when working in this directory to maintain consistency across conversations and models.

## 1. Context Gathering (Start Here)
- ALWAYS read `README.md` for project context.
- ALWAYS check `TODO.md` to see what was recently completed and what the current active task is.
- ALWAYS review `ARCHITECTURE.md` before creating new files or modifying core logic to ensure it aligns with the system design.

## 2. Coding Standards
- **Language/Framework Rules:** 
  - Use Python 3.11+.
  - Use FastAPI for the backend.
  - Use Async SQLite (`aiosqlite`) with WAL mode for the database.
  - Use Pydantic for data validation.
  - Use Ruff for linting and formatting (`line-length = 100`).
  - Use pytest for testing (aiming for 75%+ coverage).
- **Security:** NEVER commit `.env` or any secret keys. Always use `.env.example` for dummy values.
- **Modularity:** Keep functions small and modular. Do not dump all code into a single file. Follow the established directory structure in `ARCHITECTURE.md`.
- **Comments:** Add descriptive docstrings and comments for complex logic. Do not remove existing comments unless instructed.

## 3. Modifying State
- When you complete a task, you MUST update `TODO.md` by moving the task to the 'Completed' section and updating the 'Current Status'.
- Do not make massive sweeping changes across multiple directories without explicit user approval.

## 4. Environment
- The OS is Windows. Use PowerShell commands.
- A virtual environment is located at `.\venv`. Always ensure dependencies are installed here.
