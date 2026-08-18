from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
import os
import pickle
import re
import string
import time
import warnings

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

from src.data_task.data_preprocessing import preprocess_text


# ============================================================
# Configuration
# ============================================================

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

MODELS_DIR = PROJECT_ROOT / "models"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
LOCAL_MODEL_PATH = MODELS_DIR / "model.pkl"

TEMPLATES_DIR = BASE_DIR / "templates"

MODEL_NAME = os.getenv("MODEL_NAME", "my_model")

REPO_OWNER = "Santosh-Chapagain"
REPO_NAME = "Movie_Sentiment_Analysis"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow",
)

DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")


# ============================================================
# Prometheus metrics
# ============================================================

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count",
    "Total number of requests to the app",
    ["method", "endpoint"],
    registry=registry,
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Latency of requests in seconds",
    ["endpoint"],
    registry=registry,
)

PREDICTION_COUNT = Counter(
    "model_prediction_count",
    "Count of predictions for each class",
    ["prediction"],
    registry=registry,
)


# ============================================================
# FastAPI setup
# ============================================================

templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
)


# ============================================================
# NLTK
# ============================================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ============================================================
# Global model state
# ============================================================

MODEL: Optional[Any] = None
VECTORIZER: Optional[Any] = None
STARTUP_ERROR: Optional[str] = None


# ============================================================
# Text preprocessing
# ============================================================

def remove_stop_words(text: str) -> str:
    """Remove stop words from text."""
    words = [
        word
        for word in str(text).split()
        if word not in stop_words
    ]

    return " ".join(words)


def removing_numbers(text: str) -> str:
    """Remove numbers from text."""
    return "".join(
        char for char in text
        if not char.isdigit()
    )


def lower_case(text: str) -> str:
    """Convert text to lowercase."""
    return " ".join(
        word.lower()
        for word in text.split()
    )


def removing_punctuations(text: str) -> str:
    """Remove punctuation from text."""

    text = re.sub(
        r"[%s]" % re.escape(string.punctuation),
        " ",
        text,
    )

    text = text.replace("؛", "")

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def removing_urls(text: str) -> str:
    """Remove URLs from text."""

    return re.compile(
        r"https?://\S+|www\.\S+"
    ).sub("", text)


def lemmatization(text: str) -> str:
    """Lemmatize text."""

    return " ".join(
        lemmatizer.lemmatize(word)
        for word in text.split()
    )


def normalize_text(text: str) -> str:
    """
    Use the same preprocessing pipeline used during training.
    """

    return preprocess_text(text)


# ============================================================
# MLflow configuration
# ============================================================

def configure_mlflow() -> None:
    """Configure MLflow."""

    mlflow.set_tracking_uri(TRACKING_URI)

    if DAGSHUB_TOKEN:

        os.environ["MLFLOW_TRACKING_USERNAME"] = REPO_OWNER
        os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN


# ============================================================
# Model registry
# ============================================================

def get_latest_model_version(
    model_name: str,
) -> Optional[str]:

    client = mlflow.MlflowClient()

    try:

        # Old MLflow stage-based registry.
        for stage in ("Production", "None"):

            latest_versions = client.get_latest_versions(
                model_name,
                stages=[stage],
            )

            if latest_versions:

                return latest_versions[0].version

    except Exception as exc:

        raise RuntimeError(
            f"Unable to access MLflow model registry: {exc}"
        ) from exc

    return None


# ============================================================
# Vectorizer
# ============================================================

def load_vectorizer() -> Any:
    """Load vectorizer from disk."""

    if not VECTORIZER_PATH.exists():

        raise FileNotFoundError(
            f"Vectorizer file not found: "
            f"{VECTORIZER_PATH}"
        )

    with open(
        VECTORIZER_PATH,
        "rb",
    ) as file:

        return pickle.load(file)


# ============================================================
# Local model
# ============================================================

def load_local_model() -> Any:
    """Load locally saved model."""

    if not LOCAL_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Local model file not found: "
            f"{LOCAL_MODEL_PATH}"
        )

    with open(
        LOCAL_MODEL_PATH,
        "rb",
    ) as file:

        return pickle.load(file)


# ============================================================
# Remote MLflow model
# ============================================================

def load_remote_model() -> Any:
    """Load model from MLflow registry."""

    model_version = get_latest_model_version(
        MODEL_NAME
    )

    if not model_version:

        raise RuntimeError(
            f"No registered model version found "
            f"for {MODEL_NAME}"
        )

    model_uri = (
        f"models:/{MODEL_NAME}/{model_version}"
    )

    return mlflow.pyfunc.load_model(
        model_uri
    )


# ============================================================
# Model loading strategy
# ============================================================

def load_model_artifact() -> Any:
    """
    Load remote model if DagsHub is configured.

    If remote loading fails, fall back to
    the local model.
    """

    remote_errors = []

    if DAGSHUB_TOKEN:

        try:

            return load_remote_model()

        except Exception as exc:

            remote_errors.append(
                f"Remote model loading failed: {exc}"
            )

    try:

        return load_local_model()

    except Exception as exc:

        local_error = (
            f"Local model loading failed: {exc}"
        )

        if remote_errors:

            raise RuntimeError(
                "; ".join(
                    remote_errors + [local_error]
                )
            ) from exc

        raise RuntimeError(
            local_error
        ) from exc


# ============================================================
# Application bootstrap
# ============================================================

def bootstrap_artifacts() -> None:
    """
    Load vectorizer and model.

    Failure does not crash the application.
    The error is stored in STARTUP_ERROR.
    """

    global MODEL
    global VECTORIZER
    global STARTUP_ERROR

    try:

        configure_mlflow()

        VECTORIZER = load_vectorizer()

        MODEL = load_model_artifact()

        STARTUP_ERROR = None

    except Exception as exc:

        MODEL = None
        VECTORIZER = None

        STARTUP_ERROR = str(exc)


# ============================================================
# Feature generation
# ============================================================

def build_feature_frame(
    text: str,
) -> pd.DataFrame:

    if VECTORIZER is None:

        raise RuntimeError(
            "Vectorizer is not loaded"
        )

    features = VECTORIZER.transform(
        [text]
    )

    return pd.DataFrame(
        features.toarray(),
        columns=[
            str(i)
            for i in range(
                features.shape[1]
            )
        ],
    )


# ============================================================
# Prediction formatting
# ============================================================

def format_prediction(
    prediction: Any,
) -> str:

    if isinstance(
        prediction,
        np.generic,
    ):

        prediction = prediction.item()

    prediction_string = str(
        prediction
    )

    if prediction_string in {
        "1",
        "positive",
        "pos",
        "Positive",
    }:

        return "Positive"

    if prediction_string in {
        "0",
        "negative",
        "neg",
        "Negative",
    }:

        return "Negative"

    return prediction_string


# ============================================================
# Prediction
# ============================================================

def predict_sentiment(
    text: str,
) -> tuple[Any, str, str]:

    if (
        MODEL is None
        or VECTORIZER is None
    ):

        raise RuntimeError(
            STARTUP_ERROR
            or "Model artifacts are not loaded"
        )

    cleaned_text = normalize_text(
        text
    )

    features_df = build_feature_frame(
        cleaned_text
    )

    raw_prediction = MODEL.predict(
        features_df
    )[0]

    prediction_label = format_prediction(
        raw_prediction
    )

    return (
        raw_prediction,
        prediction_label,
        cleaned_text,
    )


# ============================================================
# FastAPI lifespan
# ============================================================

@asynccontextmanager
async def lifespan(
    application: FastAPI,
):

    bootstrap_artifacts()

    yield


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Movie Sentiment Analysis API",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# Request model
# ============================================================

class PredictRequest(BaseModel):
    text: str


# ============================================================
# Home page
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
def home(
    request: Request,
):

    REQUEST_COUNT.labels(
        method="GET",
        endpoint="/",
    ).inc()

    start_time = time.perf_counter()

    context = {
        "request": request,
        "result": None,
        "raw_prediction": None,
        "input_text": "",
        "model_ready": (
            MODEL is not None
            and VECTORIZER is not None
        ),
        "startup_error": STARTUP_ERROR,
    }

    response = templates.TemplateResponse(
        "index.html",
        context,
    )

    REQUEST_LATENCY.labels(
        endpoint="/"
    ).observe(
        time.perf_counter() - start_time
    )

    return response


# ============================================================
# HTML prediction endpoint
# ============================================================

@app.post(
    "/predict",
    response_class=HTMLResponse,
)
def predict(
    request: Request,
    text: str = Form(...),
):

    REQUEST_COUNT.labels(
        method="POST",
        endpoint="/predict",
    ).inc()

    start_time = time.perf_counter()

    try:

        raw_prediction, prediction_label, cleaned_text = (
            predict_sentiment(text)
        )

        PREDICTION_COUNT.labels(
            prediction=prediction_label
        ).inc()

        context = {
            "request": request,
            "result": prediction_label,
            "raw_prediction": raw_prediction,
            "input_text": text,
            "cleaned_text": cleaned_text,
            "model_ready": True,
            "startup_error": None,
        }

        response = templates.TemplateResponse(
            "index.html",
            context,
        )

        return response

    except Exception as exc:

        context = {
            "request": request,
            "result": None,
            "raw_prediction": None,
            "input_text": text,
            "cleaned_text": None,
            "model_ready": False,
            "startup_error": str(exc),
        }

        return templates.TemplateResponse(
            "index.html",
            context,
            status_code=503,
        )

    finally:

        REQUEST_LATENCY.labels(
            endpoint="/predict"
        ).observe(
            time.perf_counter() - start_time
        )


# ============================================================
# JSON prediction endpoint
# ============================================================

@app.post("/api/predict")
def api_predict(
    payload: PredictRequest,
):

    REQUEST_COUNT.labels(
        method="POST",
        endpoint="/api/predict",
    ).inc()

    start_time = time.perf_counter()

    try:

        text = payload.text.strip()

        if not text:

            raise HTTPException(
                status_code=422,
                detail="'text' is required",
            )

        (
            raw_prediction,
            prediction_label,
            cleaned_text,
        ) = predict_sentiment(text)

        PREDICTION_COUNT.labels(
            prediction=prediction_label
        ).inc()

        return JSONResponse(
            {
                "prediction": prediction_label,
                "raw_prediction": str(
                    raw_prediction
                ),
                "cleaned_text": cleaned_text,
            }
        )

    finally:

        REQUEST_LATENCY.labels(
            endpoint="/api/predict"
        ).observe(
            time.perf_counter() - start_time
        )


# ============================================================
# Health endpoint
# ============================================================

@app.get("/health")
def health():

    model_ready = (
        MODEL is not None
        and VECTORIZER is not None
    )

    return {
        "status": (
            "ok"
            if model_ready
            else "degraded"
        ),
        "model_ready": MODEL is not None,
        "vectorizer_ready": (
            VECTORIZER is not None
        ),
        "startup_error": STARTUP_ERROR,
    }


# ============================================================
# Prometheus metrics
# ============================================================

@app.get("/metrics")
def metrics():

    return Response(
        content=generate_latest(
            registry
        ),
        media_type=CONTENT_TYPE_LATEST,
    )
