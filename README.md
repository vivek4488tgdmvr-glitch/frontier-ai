# Frontier AI Monorepo

This repository combines two related projects in a single root repo:

- `backend/` — the Python/Streamlit Latent Frontier Lab app
- `frontend/` — the frontend project for the Frontier AI UI

## Project summary

The backend project is an interactive educational explainer for the DataForge 2026 Pathway Track: Explain the Frontier brief. It models a fixed-size latent state that is refined over recurrence to infer a task rule from demonstrations and solve a new query without emitting a verbal chain-of-thought.

## Repository structure

```text
.
├── backend/
├── frontend/
├── .gitignore
├── README.md
└── .git/
```

## Run the backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Optional Docker run:

```bash
docker build -t latent-frontier-lab .
docker run -p 8501:8501 latent-frontier-lab
```

## Run the frontend

```bash
cd frontend
npm install
npm run dev
```

## Tests for the backend

```bash
cd backend
python -m unittest discover -s tests -v
```

## Notes

This repo keeps both applications in one Git repository while preserving them as separate folders, so they can be developed and deployed independently.
