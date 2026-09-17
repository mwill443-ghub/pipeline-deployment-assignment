"""Fits the customer-similarity pipeline and dumps a deployable bundle."""

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors

from pipeline_def import build_pipeline

HERE = Path(__file__).parent
DATA_PATH = HERE / "data" / "cleaned_dataset.csv"
BUNDLE_PATH = HERE / "model_bundle.joblib"

FEATURE_COLUMNS = ["Age", "State", "Income", "Purchases", "LastPurchaseDate", "Review"]


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df["Review"] = df["Review"].fillna("")
    X = df[FEATURE_COLUMNS]

    pipeline = build_pipeline()
    matrix = pipeline.fit_transform(X)

    nn_model = NearestNeighbors(n_neighbors=min(5, len(df)), metric="euclidean")
    nn_model.fit(matrix)

    customers = df[["CustomerID", *FEATURE_COLUMNS]].to_dict(orient="records")
    customers = [
        {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in record.items()}
        for record in customers
    ]

    bundle = {
        "pipeline": pipeline,
        "nn_model": nn_model,
        "matrix": matrix,
        "customers": customers,
        "metadata": {
            "steps": [name for name, _ in pipeline.steps],
            "built_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "n_customers": len(df),
            "feature_columns": FEATURE_COLUMNS,
            "matrix_shape": list(matrix.shape),
        },
    }

    joblib.dump(bundle, BUNDLE_PATH)
    print(f"Saved bundle to {BUNDLE_PATH}")
    print(f"  customers: {len(df)}, matrix shape: {matrix.shape}")
    print(f"  reference date learned: {pipeline.named_steps['preprocess'].named_transformers_['date'].named_steps['recency'].reference_date_}")


if __name__ == "__main__":
    main()
