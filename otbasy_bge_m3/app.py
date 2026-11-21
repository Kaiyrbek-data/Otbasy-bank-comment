"""FastAPI application for housing-queue FAQ retrieval."""

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from retriever import ROOT, FAQRetriever


def load_settings():
    settings_path = ROOT / "config.yaml"
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError(f"Некорректная конфигурация: {settings_path}")

    required_sections = ("model", "projection", "retrieval", "confidence", "logging")
    if any(not isinstance(settings.get(section), dict) for section in required_sections):
        raise ValueError("config.yaml должен содержать model, projection, retrieval, confidence и logging.")

    retrieval = settings["retrieval"]
    confidence = settings["confidence"]
    top_k = retrieval.get("top_k")
    low = confidence.get("low_threshold")
    high = confidence.get("high_threshold")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 10:
        raise ValueError("retrieval.top_k должен быть целым числом от 1 до 10.")
    if not isinstance(retrieval.get("dedupe"), bool):
        raise ValueError("retrieval.dedupe должен быть логическим значением.")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (low, high)):
        raise ValueError("Пороги confidence должны быть числами.")
    if not -1 <= low < high <= 1:
        raise ValueError("Пороги confidence должны удовлетворять -1 <= low < high <= 1.")
    if not isinstance(settings.get("app_version"), str) or not settings["app_version"]:
        raise ValueError("app_version должен быть непустой строкой.")

    return settings


SETTINGS = load_settings()
LOG_LOCK = Lock()


def validate_model_settings(settings, model_config):
    expected = (
        (settings["model"].get("name"), model_config.get("model"), "model.name"),
        (settings["model"].get("revision"), model_config.get("revision"), "model.revision"),
        (settings["projection"].get("dimension"), model_config.get("dimension"), "projection.dimension"),
        (settings["projection"].get("rank"), model_config.get("rank"), "projection.rank"),
    )
    mismatches = [name for configured, actual, name in expected if configured != actual]
    if mismatches:
        raise ValueError(f"config.yaml не совпадает с model.json: {', '.join(mismatches)}")


def write_query_log(settings, question, top_match, confidence):
    relative_path = Path(settings["logging"]["jsonl_path"])
    log_path = relative_path if relative_path.is_absolute() else ROOT / relative_path
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question_length": len(question),
        "top1_id": top_match.get("id") if top_match else None,
        "cosine": top_match.get("cosine") if top_match else None,
        "confidence": confidence,
    }
    with LOG_LOCK:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


@asynccontextmanager
async def lifespan(app):
    model_config = json.loads((ROOT / "model.json").read_text(encoding="utf-8"))
    validate_model_settings(SETTINGS, model_config)
    retriever = FAQRetriever()
    app.state.settings = SETTINGS
    app.state.retriever = retriever
    yield


app = FastAPI(
    title="Поиск ответов об Отбасы банке",
    version=SETTINGS["app_version"],
    lifespan=lifespan,
)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="По умолчанию используется retrieval.top_k из config.yaml.",
    )
    dedupe: bool | None = Field(
        default=None,
        description="По умолчанию используется retrieval.dedupe из config.yaml (true).",
    )
    low_threshold: float | None = Field(
        default=None,
        ge=-1,
        le=1,
        description="Переопределяет confidence.low_threshold из config.yaml.",
    )
    high_threshold: float | None = Field(
        default=None,
        ge=-1,
        le=1,
        description="Переопределяет confidence.high_threshold из config.yaml.",
    )


@app.get("/", include_in_schema=False)
def demo():
    return FileResponse(ROOT / "demo.html", media_type="text/html")


@app.get("/health")
def health(request: Request):
    settings = request.app.state.settings
    return {
        "status": "ready",
        "model": settings["model"]["name"],
        "app_version": settings["app_version"],
        "index_version": settings["index_version"],
        "rows": request.app.state.retriever.config["rows"],
    }


@app.get("/topics")
def topics(request: Request):
    retriever = request.app.state.retriever
    counts = {}
    for document in retriever.docs:
        topic = document.get("topic", "Другое")
        counts[topic] = counts.get(topic, 0) + 1
    return [
        {"name": name, "count": count}
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


@app.post("/search")
def search(body: Question, request: Request):
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="Введите вопрос.")

    settings = request.app.state.settings
    retrieval = settings["retrieval"]
    confidence_settings = settings["confidence"]
    low_threshold = (
        body.low_threshold
        if body.low_threshold is not None
        else confidence_settings["low_threshold"]
    )
    high_threshold = (
        body.high_threshold
        if body.high_threshold is not None
        else confidence_settings["high_threshold"]
    )
    if low_threshold >= high_threshold:
        raise HTTPException(
            status_code=422,
            detail="low_threshold должен быть меньше high_threshold.",
        )

    matches = request.app.state.retriever.search(
        body.question,
        top_k=body.top_k if body.top_k is not None else retrieval["top_k"],
        dedupe=body.dedupe if body.dedupe is not None else retrieval["dedupe"],
    )
    top_match = matches[0] if matches else None
    cosine = top_match["cosine"] if top_match else None

    if cosine is None or cosine < low_threshold:
        confidence = "low"
        found = False
        visible_matches = []
    elif cosine >= high_threshold:
        confidence = "high"
        found = True
        visible_matches = matches
    else:
        confidence = "medium"
        found = True
        visible_matches = matches

    write_query_log(settings, body.question, top_match, confidence)
    return {
        "matches": visible_matches,
        "found": found,
        "confidence": confidence,
    }
