"""FastAPI adapter for the currently approved champion."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .service import ModelService

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "logistic_champion.joblib"

app = FastAPI(title="Credit Risk Champion API", version="0.1.0")


class Application(BaseModel):
    model_config = ConfigDict(extra="forbid")

    AMT_INCOME_TOTAL: float = Field(gt=0)
    AMT_CREDIT: float = Field(gt=0)
    AMT_ANNUITY: float = Field(gt=0)
    DAYS_BIRTH: int = Field(lt=0)
    DAYS_EMPLOYED: int
    EXT_SOURCE_1: float | None = Field(default=None, ge=0, le=1)
    EXT_SOURCE_2: float | None = Field(default=None, ge=0, le=1)
    EXT_SOURCE_3: float | None = Field(default=None, ge=0, le=1)
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str


@lru_cache(maxsize=1)
def get_service() -> ModelService:
    model_path = Path(os.environ.get("CREDIT_RISK_MODEL_PATH", DEFAULT_MODEL_PATH))
    return ModelService(model_path)


@app.get("/health")
def health() -> dict[str, str]:
    try:
        service = get_service()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc
    return {"status": "ok", "champion": service.model_name}


@app.post("/predict")
def predict(application: Application) -> dict[str, object]:
    try:
        return get_service().score(application.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Scoring failed") from exc

