"""FastAPI service exposing the fitted customer-similarity pipeline."""

import re
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import pipeline_def  # noqa: F401  (required so joblib can unpickle the custom transformer)

BUNDLE_PATH = Path(__file__).parent / "model_bundle.joblib"

try:
    _bundle = joblib.load(BUNDLE_PATH)
except Exception:
    _bundle = None

pipeline = _bundle["pipeline"] if _bundle else None
nn_model = _bundle["nn_model"] if _bundle else None
customers = _bundle["customers"] if _bundle else None
metadata = _bundle["metadata"] if _bundle else None

app = FastAPI(title="Customer Similarity API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_bundle():
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded")


DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class CustomerProfile(BaseModel):
    Age: int = Field(ge=0, le=120)
    State: str = Field(pattern=r"^[A-Za-z]{2}$")
    Income: float = Field(ge=0)
    Purchases: int = Field(ge=0)
    LastPurchaseDate: Optional[str] = None
    Review: str = Field(default="", max_length=1000)

    @field_validator("State")
    @classmethod
    def uppercase_state(cls, value: str) -> str:
        return value.upper()

    @field_validator("LastPurchaseDate")
    @classmethod
    def validate_date(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not DATE_PATTERN.match(value):
            raise ValueError("LastPurchaseDate must be in YYYY-MM-DD format")
        return value


@app.get("/health")
def health():
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded")
    return {"status": "ok"}


@app.get("/metadata")
def get_metadata():
    _require_bundle()
    return metadata


@app.get("/customers")
def list_customers():
    _require_bundle()
    return customers


@app.post("/similar-customers")
def similar_customers(profile: CustomerProfile, k: int = 5):
    _require_bundle()
    row = pd.DataFrame([profile.model_dump()])
    vector = pipeline.transform(row)

    k = max(1, min(k, len(customers)))
    distances, indices = nn_model.kneighbors(vector, n_neighbors=k)

    return [
        {**customers[idx], "distance": float(dist)}
        for dist, idx in zip(distances[0], indices[0])
    ]
