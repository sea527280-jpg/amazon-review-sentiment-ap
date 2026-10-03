from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_reports_loaded_model() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["status"] == "ok"


def test_prediction_returns_sentiment_and_probabilities() -> None:
    response = client.post(
        "/predict",
        json={
            "reviewText": (
                "Absolutely wonderful and entertaining game. I love it and "
                "highly recommend it."
            )
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == 1
    assert result["sentiment"] == "positive"
    assert set(result["probabilities"]) == {"negative", "positive"}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6
    assert result["confidence"] == result["probabilities"]["positive"]


def test_prediction_rejects_blank_review() -> None:
    response = client.post("/predict", json={"reviewText": "   "})

    assert response.status_code == 422


def test_prediction_rejects_extra_fields() -> None:
    response = client.post(
        "/predict",
        json={"reviewText": "Great app.", "Positive": 1},
    )

    assert response.status_code == 422


def test_prediction_reports_unavailable_model(monkeypatch) -> None:
    monkeypatch.setattr(app.state, "model", None)
    monkeypatch.setattr(app.state, "model_error", "test model load error")

    response = client.post("/predict", json={"reviewText": "Great app."})

    assert response.status_code == 503
    assert response.json()["detail"] == "test model load error"
