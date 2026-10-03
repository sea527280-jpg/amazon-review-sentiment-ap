"""Train and export the Amazon review sentiment classifier."""

from __future__ import annotations

import argparse
import csv
import io
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

DATASET_URL = (
    "https://raw.githubusercontent.com/Jimmi-Kr/"
    "Getting-started-ml-production/main/datasets/amazon_dataset.csv"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "sentiment_model.pkl"


def load_dataset(url: str) -> tuple[list[str], list[int]]:
    request = Request(url, headers={"User-Agent": "amazon-review-sentiment-api/1.0"})
    with urlopen(request, timeout=60) as response:
        csv_text = response.read().decode("utf-8-sig")

    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames is None or not {"reviewText", "Positive"}.issubset(
        reader.fieldnames
    ):
        raise ValueError("Dataset must contain 'reviewText' and 'Positive' columns.")

    reviews: list[str] = []
    labels: list[int] = []
    for row_number, row in enumerate(reader, start=2):
        review = (row.get("reviewText") or "").strip()
        label_text = (row.get("Positive") or "").strip()
        if not review or label_text not in {"0", "1"}:
            raise ValueError(
                f"Invalid review text or Positive label on CSV row {row_number}."
            )
        reviews.append(review)
        labels.append(int(label_text))

    if len(reviews) < 2 or len(set(labels)) != 2:
        raise ValueError("Dataset must contain reviews from both sentiment classes.")
    return reviews, labels


def build_model() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=30_000,
                    min_df=2,
                    max_df=0.98,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    max_features=20_000,
                    min_df=2,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


def train_model(data_url: str, output_path: Path) -> dict[str, float]:
    reviews, labels = load_dataset(data_url)
    x_train, x_test, y_train, y_test = train_test_split(
        reviews,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    evaluation_model = build_model()
    evaluation_model.fit(x_train, y_train)
    predictions = evaluation_model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
    }

    final_model = build_model()
    final_model.fit(reviews, labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(
        prefix="sentiment-model-",
        suffix=".pkl",
        dir=output_path.parent,
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    try:
        joblib.dump(final_model, temporary_path, compress=3)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-url", default=DATASET_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    metrics = train_model(args.data_url, args.output)
    print(
        f"Saved trained model to {args.output}\n"
        f"Holdout accuracy: {metrics['accuracy']:.4f}\n"
        f"Holdout macro F1: {metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
