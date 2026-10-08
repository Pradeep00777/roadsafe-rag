# Deploy RoadSafe RAG to Render (free, no card)

Render's free web service has 512 MB RAM and a fractional CPU, which is too small for PyTorch. `Dockerfile.render` swaps `sentence-transformers` for `fastembed` (the same MiniLM model, run with ONNX) so the app can fit. Free services sleep after 15 minutes without traffic and take about a minute to wake up. Check Render's pricing page before you start, because free-tier terms change.

## 1. Push the patch
Copy `Dockerfile.render`, `requirements-render.txt`, `src/roadsafe_rag/embeddings.py` and `docs/DEPLOY_RENDER.md` into your repo, then commit and push to GitHub.

## 2. Create the service
1. Sign up at render.com with your GitHub account.
2. **New, Web Service**, then pick the `roadsafe-rag` repo.
3. Settings:
   - Language: **Docker**
   - Region: **Singapore** (closest to India)
   - Instance type: **Free**
   - Dockerfile path: `./Dockerfile.render`
   - Health check path: `/health`
4. Environment variables:
   - `GROQ_API_KEY`: a **new** key (revoke any key that was pasted elsewhere)
   - `GROQ_MODEL`: the exact model used in your evaluation
   - `RATE_LIMIT_PER_MIN`: `10`
5. **Create Web Service**. The first build takes about 10-15 minutes.

## 3. Test
Open the `onrender.com` URL, check `/health` shows `"ready": true`, ask a real question and an off-topic one. Open the demo yourself before sending an application so the service is awake.

## 4. Make your numbers match the demo
The demo uses fastembed, your README results used sentence-transformers. Re-run the evaluation with the same embedder and update the README:

```bash
pip install fastembed
EMBEDDER=fastembed INDEX_DIR=data/index_fastembed python -m roadsafe_rag.ingest --pdf data/raw/road-accidents-in-india-2024.pdf
EMBEDDER=fastembed INDEX_DIR=data/index_fastembed python -m roadsafe_rag.evaluate
```
(On Windows use `set EMBEDDER=fastembed` and `set INDEX_DIR=data/index_fastembed` first.) Report the new table, or both, and say which embedder each used.

## If it fails
- **Out of memory in the logs:** the free 512 MB is not enough. Do not pay for more; use a screen recording instead (GIF in the README).
- **Build fails at the download step:** the report link on OpenCity may have changed, so update `scripts/download_data.py`.
- **Model name not found:** set `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5` in `Dockerfile.render`, rebuild, and re-run the evaluation.
