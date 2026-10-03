# Frontier AI Monorepo

This repository contains two related projects in separate folders:

- `backend/` — the Python/Streamlit Latent Frontier Lab app
- `frontend/` — the React/Vite frontend repo cloned from the GitHub project

## Structure

```text
.
├── backend/
├── frontend/
├── .gitignore
├── README.md
└── .git/
```

## Run locally

### Backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Notes

This root repo keeps both projects under the same Git repository while preserving them as separate folders.
