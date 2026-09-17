"""Custom transformer and Pipeline factory for the customer-similarity model."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["Age", "Income", "Purchases"]
CATEGORICAL_FEATURES = ["State"]
DATE_FEATURE = ["LastPurchaseDate"]
TEXT_FEATURE = "Review"


class DaysSinceLastPurchaseTransformer(BaseEstimator, TransformerMixin):
    """Converts a date column into "days since last purchase".

    The reference date and the fill-in value for missing dates are both
    learned from the training data at fit time, so this transformer carries
    real state: re-instantiating it without the fitted attributes (or
    fitting it on a different slice of data) produces different numbers for
    the same input dates.
    """

    def __init__(self, date_format: str = "%Y-%m-%d"):
        self.date_format = date_format

    def fit(self, X, y=None):
        dates = self._to_datetime(X)
        self.reference_date_ = dates.max()
        day_deltas = (self.reference_date_ - dates).dt.days
        self.fallback_days_ = float(day_deltas.median(skipna=True))
        return self

    def transform(self, X):
        dates = self._to_datetime(X)
        day_deltas = (self.reference_date_ - dates).dt.days.astype(float)
        day_deltas = day_deltas.fillna(self.fallback_days_)
        return day_deltas.to_numpy().reshape(-1, 1)

    def _to_datetime(self, X):
        column = pd.DataFrame(X).iloc[:, 0]
        return pd.to_datetime(column, format=self.date_format, errors="coerce")


def build_pipeline(n_components: int = 8) -> Pipeline:
    """Builds the (unfitted) preprocessing + dimensionality-reduction pipeline."""

    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    date_pipeline = Pipeline(
        [
            ("recency", DaysSinceLastPurchaseTransformer()),
            ("scale", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("date", date_pipeline, DATE_FEATURE),
            ("text", TfidfVectorizer(max_features=50), TEXT_FEATURE),
        ]
    )

    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("reduce", TruncatedSVD(n_components=n_components, random_state=42)),
        ]
    )
