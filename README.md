# Amazon Review Sentiment API

This API classifies Amazon app review text as **positive** (`1`) or **negative** (`0`) using a TF-IDF and logistic-regression model trained on the dataset's `reviewText` and `Positive` columns.

## Example request

```json
{
  "reviewText": "This is a great game, lots of fun, and I highly recommend it."
}
```

Send that JSON to `POST /predict`. The response includes the predicted label, sentiment, confidence, and probabilities for both classes.

## Run locally

Python 3.12 or newer is required.

```bash
python -m venv .venv
# Activate the virtual environment, then:
pip install -r requirements.txt
python train.py
uvicorn main:app --reload
```

The training command downloads the CSV from the configured dataset URL and writes `models/sentiment_model.pkl`. The API is available at `http://127.0.0.1:8000`; interactive docs are at `/docs`.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"reviewText":"This is a great game and I love playing it."}'
```

Run API tests with `pytest`.

## Deployment

`render.yaml` defines a Render web service. Connect the repository in Render and deploy the Blueprint; Render uses `/health` as its health check.
