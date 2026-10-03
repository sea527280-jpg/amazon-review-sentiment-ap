"""FastAPI application for Amazon review sentiment prediction."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "sentiment_model.pkl"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))


def load_model() -> tuple[object | None, str | None]:
    try:
        model = joblib.load(MODEL_PATH)
    except (
        OSError,
        EOFError,
        ImportError,
        pickle.PickleError,
        AttributeError,
        ValueError,
        TypeError,
    ) as exc:
        return None, f"Could not load model at {MODEL_PATH}: {exc}"

    if not callable(getattr(model, "predict", None)) or not callable(
        getattr(model, "predict_proba", None)
    ):
        return None, "Loaded model must provide predict and predict_proba methods."
    return model, None


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    reviewText: str = Field(
        min_length=1,
        max_length=10_000,
        description="The Amazon review text to classify.",
    )

    @field_validator("reviewText")
    @classmethod
    def reject_blank_review(cls, value: str) -> str:
        if not value:
            raise ValueError("reviewText must contain non-whitespace characters.")
        return value


class PredictionResponse(BaseModel):
    prediction: int
    sentiment: str
    confidence: float
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_error: str | None = None


app = FastAPI(
    title="Amazon Review Sentiment API",
    description="Classifies Amazon app review text as positive or negative.",
    version="1.0.0",
)
app.state.model, app.state.model_error = load_model()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model_loaded = app.state.model is not None
    return HealthResponse(
        status="ok" if model_loaded else "unavailable",
        model_loaded=model_loaded,
        model_error=app.state.model_error,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    model = app.state.model
    if model is None:
        detail = app.state.model_error or "The sentiment model is not loaded."
        raise HTTPException(status_code=503, detail=detail)

    try:
        prediction = int(model.predict([request.reviewText])[0])
        probabilities = model.predict_proba([request.reviewText])[0]
    except (ValueError, TypeError, AttributeError, RuntimeError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}"
        ) from exc

    class_probabilities = {
        int(class_label): float(probability)
        for class_label, probability in zip(model.classes_, probabilities, strict=True)
    }
    if prediction not in {0, 1} or set(class_probabilities) != {0, 1}:
        raise HTTPException(
            status_code=500,
            detail="The sentiment model returned an unsupported class.",
        )

    return PredictionResponse(
        prediction=prediction,
        sentiment="positive" if prediction == 1 else "negative",
        confidence=class_probabilities[prediction],
        probabilities={
            "negative": class_probabilities[0],
            "positive": class_probabilities[1],
        },
    )
