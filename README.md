# Homework 4 — Fitted Pipeline as a Deployed Service

A customer-similarity search API. A fitted scikit-learn `Pipeline` turns a
customer profile (age, state, income, purchases, last purchase date, review
text) into an 8-dimensional feature vector; a fitted `NearestNeighbors`
index finds the most similar customers from the training set. It's served
by FastAPI, deployed on Modal, and called from a small static frontend on
Vercel.

## Why it needs a fitted bundle, not just pipeline code

Three pieces of state are learned from the training data and would be wrong
if the pipeline were rebuilt from scratch at boot instead of loaded from
the dump:

- `DaysSinceLastPurchaseTransformer.reference_date_` / `fallback_days_`
  (`pipeline_def.py`) — the "today" used to compute recency, and the
  fill-in value for missing dates, are both derived from the training data.
- The `TfidfVectorizer` vocabulary and IDF weights learned from the 76
  `Review` strings.
- The `TruncatedSVD` components and the `NearestNeighbors` index, both fit
  on the transformed training matrix.

## Files

| File | Purpose |
| --- | --- |
| `pipeline_def.py` | `DaysSinceLastPurchaseTransformer` (custom, stateful) + `build_pipeline()` |
| `train.py` | Fits the pipeline on `data/cleaned_dataset.csv`, fits a `NearestNeighbors` index on the resulting matrix, and dumps everything to `model_bundle.joblib` |
| `api.py` | FastAPI app; loads the bundle once at import time and exposes it |
| `modal_app.py` | Wraps the FastAPI app as a Modal ASGI app for deployment |
| `frontend/` | Static HTML/CSS/JS UI that calls the deployed API |
| `postman_collection.json` / `postman_environment.json` | Importable Postman collection + environment pointed at the Modal URL |

## The bundle (`model_bundle.joblib`)

```python
{
    "pipeline": Pipeline,          # fitted: preprocess (ColumnTransformer) -> TruncatedSVD
    "nn_model": NearestNeighbors,  # fitted on the transformed matrix
    "matrix": np.ndarray,          # (n_customers, 8) transformed training features
    "customers": list[dict],       # original rows, used to render results
    "metadata": {
        "steps": [...],
        "built_at": "...",
        "sklearn_version": "...",
        "n_customers": 76,
        "feature_columns": [...],
        "matrix_shape": [76, 8],
    },
}
```

## Running locally

```bash
uv run python train.py
uv run uvicorn api:app --reload --port 8000
```

Then open http://localhost:8000/docs.

```bash
curl -X POST http://127.0.0.1:8123/similar-customers \
  -H "Content-Type: application/json" \
  -d '{"Age": 45, "State": "CA", "Income": 50000, "Purchases": 10, "LastPurchaseDate": "2021-07-15", "Review": "Loved it!!!"}'
```

## Deploying to Modal

```bash
modal deploy modal_app.py
```

Deployed at: **https://mwill443-ghub--pipeline-deployment-assignment-web.modal.run**

Endpoints:
- `GET /health`
- `GET /metadata`
- `GET /customers`
- `POST /similar-customers?k=5` — body: `{Age, State, Income, Purchases, LastPurchaseDate?, Review?}`

## Deploying the frontend to Vercel

This repo is connected to Vercel via GitHub — pushing to `master` triggers
an automatic deploy, with the Vercel project's Root Directory set to
`frontend/`.

The frontend's API base URL field defaults to the Modal URL above and can
be overridden at runtime without redeploying.

Deployed at: **https://pipeline-deployment-assignment.vercel.app**

## Input validation and failure modes

- Every field on `POST /similar-customers` is bounds-checked with Pydantic
  (`Age` 0-120, `Income`/`Purchases` non-negative, `State` a 2-letter code,
  `LastPurchaseDate` must be `YYYY-MM-DD` if present) — invalid input
  returns `422` with a field-by-field explanation.
- If `model_bundle.joblib` is missing or fails to unpickle at import time,
  every endpoint returns `503` instead of crashing or returning `500`.

## Testing with Postman

1. Import `postman_collection.json` and `postman_environment.json` into
   Postman.
2. Select the "Customer Similarity API - Modal" environment.
3. Run "Health check", "Pipeline metadata", "List customers", "Find similar
   customers (valid input)" (expects `200`), and "Find similar customers
   (invalid input)" (expects `422`). Each request carries `pm.test`
   assertions on status code and response shape.
