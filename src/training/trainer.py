"""
Model training with reproducible experiments and MLflow-compatible logging.
Author: Prateek Gaur
"""

import os
import json
import time
import hashlib
import logging
import pickle
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics         import (classification_report, roc_auc_score,
                                     f1_score, precision_score, recall_score)
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Fully reproducible experiment configuration."""
    experiment_name:  str  = "default_experiment"
    model_type:       str  = "random_forest"        # random_forest | gradient_boosting | svm | logistic
    target_col:       str  = "target"
    test_size:        float = 0.2
    val_size:         float = 0.1
    random_seed:      int  = 42
    cv_folds:         int  = 5
    model_params:     Dict = field(default_factory=dict)
    feature_cols:     list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def run_id(self) -> str:
        """Unique hash identifying this exact configuration."""
        raw = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()[:10]


@dataclass
class ExperimentResult:
    run_id:        str
    config:        dict
    metrics:       dict
    model_path:    str
    duration_s:    float
    timestamp:     str
    feature_importance: dict = field(default_factory=dict)


MODEL_REGISTRY = {
    "random_forest": lambda p: RandomForestClassifier(
        n_estimators=p.get("n_estimators", 200),
        max_depth=p.get("max_depth", None),
        class_weight="balanced",
        random_state=42, n_jobs=-1,
        **{k: v for k, v in p.items() if k not in ["n_estimators","max_depth"]},
    ),
    "gradient_boosting": lambda p: GradientBoostingClassifier(
        n_estimators=p.get("n_estimators", 100),
        learning_rate=p.get("learning_rate", 0.1),
        max_depth=p.get("max_depth", 3),
        random_state=42,
    ),
    "svm": lambda p: SVC(
        C=p.get("C", 1.0),
        kernel=p.get("kernel", "rbf"),
        probability=True,
        class_weight="balanced",
        random_state=42,
    ),
    "logistic": lambda p: LogisticRegression(
        C=p.get("C", 1.0),
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    ),
}


class ExperimentTracker:
    """Lightweight experiment tracker (MLflow-compatible output format)."""

    def __init__(self, tracking_dir: str = "experiments"):
        self.tracking_dir = tracking_dir
        os.makedirs(tracking_dir, exist_ok=True)
        self._log_path = os.path.join(tracking_dir, "runs.jsonl")

    def log(self, result: ExperimentResult):
        entry = {
            "run_id":     result.run_id,
            "timestamp":  result.timestamp,
            "duration_s": result.duration_s,
            "metrics":    result.metrics,
            "model_path": result.model_path,
            "config":     result.config,
        }
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(f"Run {result.run_id} logged → {self._log_path}")

    def load_all(self) -> pd.DataFrame:
        if not os.path.exists(self._log_path):
            return pd.DataFrame()
        records = []
        with open(self._log_path) as f:
            for line in f:
                records.append(json.loads(line))
        return pd.json_normalize(records)

    def best_run(self, metric: str = "metrics.f1_macro") -> Optional[dict]:
        df = self.load_all()
        if df.empty or metric not in df.columns:
            return None
        return df.loc[df[metric].idxmax()].to_dict()


class ModelTrainer:
    """
    Reproducible model training with cross-validation and experiment logging.
    """

    def __init__(self, config: ExperimentConfig, tracker: Optional[ExperimentTracker] = None):
        self.config  = config
        self.tracker = tracker or ExperimentTracker()
        np.random.seed(config.random_seed)

    def _build_pipeline(self) -> Pipeline:
        model = MODEL_REGISTRY[self.config.model_type](self.config.model_params)
        steps = [("scaler", StandardScaler()), ("model", model)]
        return Pipeline(steps)

    def train(
        self,
        df: pd.DataFrame,
        model_dir: str = "models",
    ) -> ExperimentResult:
        cfg = self.config
        t0  = time.time()

        X = df[cfg.feature_cols] if cfg.feature_cols else df.drop(columns=[cfg.target_col])
        y = df[cfg.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_seed, stratify=y
        )

        pipeline = self._build_pipeline()
        cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_seed)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)

        pipeline.fit(X_train, y_train)
        y_pred  = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

        metrics = {
            "f1_macro":   float(f1_score(y_test, y_pred, average="macro")),
            "precision":  float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "recall":     float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std":  float(cv_scores.std()),
            "roc_auc":    float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None,
            "n_train":    len(X_train),
            "n_test":     len(X_test),
        }

        # Feature importance
        feat_imp = {}
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            feat_imp = dict(zip(X.columns, model.feature_importances_.tolist()))

        # Save model
        run_id = cfg.run_id()
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{cfg.experiment_name}_{run_id}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

        result = ExperimentResult(
            run_id=run_id,
            config=cfg.to_dict(),
            metrics=metrics,
            model_path=model_path,
            duration_s=time.time() - t0,
            timestamp=pd.Timestamp.now().isoformat(),
            feature_importance=feat_imp,
        )
        self.tracker.log(result)

        logger.info(
            f"Run {run_id} | F1={metrics['f1_macro']:.4f} "
            f"CV={metrics['cv_f1_mean']:.4f}±{metrics['cv_f1_std']:.4f} "
            f"({metrics['duration_s']:.1f}s)"
        )
        return result
