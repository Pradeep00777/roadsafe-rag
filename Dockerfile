FROM python:3.12-slim

ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src \
    EMBEDDING_MODEL=${EMBEDDING_MODEL} INDEX_DIR=/app/data/index

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Bake the embedding model into the image so the container starts offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

COPY src ./src
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "roadsafe_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
