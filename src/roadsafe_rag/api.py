"""FastAPI service.

    uvicorn roadsafe_rag.api:app --reload
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .factory import build_pipeline
from .guardrails import MAX_QUESTION_CHARS
from .llm import ExtractiveLLM
from .ui import INDEX_HTML

logger = logging.getLogger("roadsafe_rag")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)


class CitationOut(BaseModel):
    id: int
    source: str
    page: int
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    refused: bool
    reason: str | None
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.pipeline = build_pipeline()
        logger.info("Pipeline ready")
    except Exception as exc:  # index missing or embedder mismatch
        app.state.pipeline = None
        logger.error("Pipeline failed to load: %s", exc)
    yield


app = FastAPI(title="RoadSafe RAG", version="0.1.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/health")
def health() -> dict:
    ready = app.state.pipeline is not None
    return {"status": "ok" if ready else "index_not_loaded", "ready": ready}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> dict:
    pipeline = app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Index not loaded. Run ingestion first.")
    try:
        result = pipeline.answer(req.question)
    except RuntimeError as exc:  # LLM provider failure, graceful fallback
        logger.warning("Primary LLM generation failed: %s; falling back to extractive answerer", exc)
        prev_llm = pipeline.llm
        try:
            pipeline.llm = ExtractiveLLM()
            result = pipeline.answer(req.question)
        finally:
            pipeline.llm = prev_llm

    logger.info("ask refused=%s reason=%s latency_ms=%s", result.refused, result.reason, result.latency_ms)
    return result.to_dict()

