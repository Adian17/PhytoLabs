# Deploy PhytoLabs on Railway

Two Railway **services** from this repo: **API** (Python/FastAPI) and **Web** (React static).

## 1. API service

1. [Railway](https://railway.app) → **New Project** → **Deploy from GitHub** → select `Adian17/PhytoLabs`.
2. Service settings:
   - **Root directory:** `/` (repo root)
   - **Config file:** `railway.toml` (auto-detected)
   - Builds from root `Dockerfile`
3. After deploy, open **Settings → Networking → Generate domain** (e.g. `phytolabs-api-production.up.railway.app`).
4. Verify: `https://<api-domain>/health` → `"models_loaded": true`.

### Use Colab-trained models (recommended)

The Docker image **bootstraps synthetic artifacts** at build time so deploy works out of the box. For real results:

1. Save in Colab: `leaf_gmm.save("gmm.joblib")`, `model.save("logreg.joblib")`.
2. Railway → API service → **Volumes** → mount at `/app/artifacts`.
3. Upload `gmm.joblib` and `logreg.joblib` into the volume (CLI or one-off deploy copy).
4. Redeploy or restart — env vars `PHYTOLABS_GMM` / `PHYTOLABS_LOGREG` already point to `/app/artifacts/`.

### API environment variables

| Variable | Default | Notes |
| --- | --- | --- |
| `PORT` | (Railway sets) | Used automatically |
| `PHYTOLABS_CORS_ORIGINS` | `*` | Set to your web app URL in production |

---

## 2. Web app service

1. Same project → **New Service** → **GitHub repo** (same repo).
2. Service settings:
   - **Root directory:** `app`
   - **Config file:** `app/railway.toml`
3. **Variables** → build-time:
   - `VITE_API_URL` = `https://<your-api-domain>` (no trailing slash)
4. **Networking** → Generate domain for the web app.
5. Open the web URL on your phone — upload a leaf photo.

---

## 3. CORS (after web is live)

On the **API** service, set:

```
PHYTOLABS_CORS_ORIGINS=https://<your-web-domain>
```

Redeploy API.

---

## Local parity

```bash
# API
phytolabs-api

# Web (proxies /api → localhost:8000)
cd app && npm run dev
```

Production build test:

```bash
cd app
VITE_API_URL=https://<api-domain> npm run build
npx vite preview
```

---

## Troubleshooting

| Issue | Fix |
| --- | --- |
| Web URL shows `{"detail":"Not Found"}` | **Wrong service deployed** — you're hitting the FastAPI API, not the React app. Set web service **Root Directory** to `app` and redeploy. Logs should show `npm`, not `python`. |
| Web shows analysis error | Check API `/health`, `VITE_API_URL`, CORS |
| Wrong predictions | Replace synthetic artifacts with Colab-trained `.joblib` on volume |
| Build slow | API image trains synthetic models once at build (~1–2 min) |
