"""
Data ingestion: REST APIs, SQL databases, and CSV/Parquet files.
Author: Prateek Gaur
"""

import os
import logging
import pandas as pd
import requests
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """Unified data ingestion from multiple sources."""

    def from_rest_api(
        self,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        auth_token: str = "",
        paginate: bool = False,
        page_size: int = 100,
    ) -> pd.DataFrame:
        """Fetch data from a REST API (with optional pagination)."""
        _headers = headers or {}
        if auth_token:
            _headers["Authorization"] = f"Bearer {auth_token}"

        all_records = []
        page = 1

        while True:
            _params = dict(params or {})
            if paginate:
                _params.update({"page": page, "page_size": page_size})

            resp = requests.get(url, params=_params, headers=_headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # Handle both list and dict responses
            records = data if isinstance(data, list) else data.get("results", data.get("data", [data]))
            all_records.extend(records)

            if not paginate or len(records) < page_size:
                break
            page += 1

        df = pd.json_normalize(all_records)
        logger.info(f"REST API ingestion: {len(df)} rows from {url}")
        return df

    def from_sql(
        self,
        connection_string: str,
        query: str,
        params: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Fetch data from SQL database (PostgreSQL, MSSQL, SQLite)."""
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        logger.info(f"SQL ingestion: {len(df)} rows")
        return df

    def from_csv(self, path: str, **kwargs) -> pd.DataFrame:
        df = pd.read_csv(path, **kwargs)
        logger.info(f"CSV ingestion: {len(df)} rows from {path}")
        return df

    def from_parquet(self, path: str) -> pd.DataFrame:
        df = pd.read_parquet(path)
        logger.info(f"Parquet ingestion: {len(df)} rows from {path}")
        return df

    def save(self, df: pd.DataFrame, path: str, format: str = "parquet"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if format == "parquet":
            df.to_parquet(path, index=False)
        elif format == "csv":
            df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} rows to {path}")


class FeatureEngineer:
    """
    Modular feature engineering for tabular ML datasets.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._fitted = False
        self._stats: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        num_cols = df.select_dtypes(include="number").columns.tolist()
        self._stats = {
            "num_cols": num_cols,
            "means":    df[num_cols].mean().to_dict(),
            "stds":     df[num_cols].std().to_dict(),
            "medians":  df[num_cols].median().to_dict(),
        }
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform()")
        df = df.copy()

        # Fill missing values
        for col in self._stats["num_cols"]:
            if col in df.columns:
                df[col] = df[col].fillna(self._stats["medians"][col])

        # Standardize numeric columns
        for col in self._stats["num_cols"]:
            if col in df.columns:
                std = self._stats["stds"][col]
                mean = self._stats["means"][col]
                if std > 0:
                    df[f"{col}_scaled"] = (df[col] - mean) / std

        # Encode categoricals
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype("category").cat.codes

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
