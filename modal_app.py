"""Deploys the customer-similarity FastAPI service to Modal."""

from pathlib import Path

import modal

app = modal.App("pipeline-deployment-assignment")

HERE = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.141.1",
        "scikit-learn==1.9.1",
        "pandas==3.0.5",
        "joblib==1.6.0",
        "numpy==2.4.6",
        "pydantic==2.13.5",
    )
    .add_local_python_source("pipeline_def", "api")
    .add_local_file(HERE / "model_bundle.joblib", "/root/model_bundle.joblib")
)


@app.function(image=image)
@modal.asgi_app()
def web():
    from api import app as fastapi_app

    return fastapi_app
