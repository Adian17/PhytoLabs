# PhytoLabs Web App

React + Vite frontend from [Figma Make](https://www.figma.com/make/UHupARiE00wxKGkWplkIKG/PhytoLabs-Design-Handoff), wired to the PhytoLabs FastAPI backend.

## Run locally (two terminals)

**Terminal 1 — API** (from repo root):

```bash
source .venv/bin/activate
pip install -e ".[api]"
phytolabs-api
```

Ensure `artifacts/gmm.joblib` and `artifacts/logreg.joblib` exist (train in Colab or run the synthetic CLI quick start).

**Terminal 2 — App**:

```bash
cd app
npm install
npm run dev
```

Open http://localhost:5173 — uploads go to `/api/analyze` (Vite proxies to `:8000`).

## Production (Railway)

1. Deploy the **API** service (Python) with artifacts mounted or baked into the image.
2. Deploy this **app** as a static site (or Vite preview server).
3. Set `VITE_API_URL=https://your-api.up.railway.app` when building the app.

```bash
VITE_API_URL=https://your-api.up.railway.app npm run build
```

## Flow

Home → Capture/Upload → Confirm → Analyzing (calls `POST /analyze`) → Results (live overlay + severity) → Treatment (placeholder; LLM later)
