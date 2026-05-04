"""
End-to-end pipeline runner: ingest → feature engineer → train → evaluate → serve.
Author: Prateek Gaur

Usage:
    python pipeline_runner.py --mode train    --config configs/pipeline_config.yaml
    python pipeline_runner.py --mode evaluate --config configs/pipeline_config.yaml
    python pipeline_runner.py --mode serve    --config configs/pipeline_config.yaml
"""

import argparse
import logging
import yaml
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from data.ingestion  import DataIngestionPipeline, FeatureEngineer
from training.trainer import ExperimentConfig, ExperimentTracker, ModelTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_train(cfg: dict):
    logger.info("=== TRAINING PIPELINE ===")
    ingestion = DataIngestionPipeline()

    # Load or generate data
    data_cfg = cfg.get("data", {})
    if data_cfg.get("source") == "csv":
        df = ingestion.from_csv(data_cfg["path"])
    elif data_cfg.get("source") == "sql":
        df = ingestion.from_sql(data_cfg["connection_string"], data_cfg["query"])
    elif data_cfg.get("source") == "api":
        df = ingestion.from_rest_api(data_cfg["url"], params=data_cfg.get("params"))
    else:
        logger.info("No data source configured — generating synthetic dataset")
        X, y = make_classification(
            n_samples=5000, n_features=20, n_informative=10,
            n_redundant=5, random_state=42
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
        df["target"] = y

    # Feature engineering
    fe = FeatureEngineer()
    df = fe.fit_transform(df)
    os.makedirs("models", exist_ok=True)
    with open("models/feature_engineer.pkl", "wb") as f:
        pickle.dump(fe, f)

    # Training
    train_cfg = cfg.get("training", {})
    exp_config = ExperimentConfig(
        experiment_name = train_cfg.get("experiment_name", "ml_pipeline"),
        model_type      = train_cfg.get("model_type", "random_forest"),
        target_col      = train_cfg.get("target_col", "target"),
        model_params    = train_cfg.get("model_params", {}),
        random_seed     = train_cfg.get("random_seed", 42),
    )

    trainer = ModelTrainer(exp_config)
    result  = trainer.train(df, model_dir="models")

    logger.info(f"Training complete. F1={result.metrics['f1_macro']:.4f}")
    return result


def run_evaluate(cfg: dict):
    logger.info("=== EVALUATION ===")
    tracker = ExperimentTracker()
    df      = tracker.load_all()
    if df.empty:
        logger.warning("No experiment runs found.")
        return
    print(df[["run_id","metrics.f1_macro","metrics.cv_f1_mean","duration_s"]].to_string(index=False))
    best = tracker.best_run("metrics.f1_macro")
    if best:
        logger.info(f"Best run: {best['run_id']}  F1={best['metrics.f1_macro']:.4f}")


def run_serve(cfg: dict):
    import uvicorn
    from serving.api import app, load_model
    serve_cfg  = cfg.get("serving", {})
    model_path = serve_cfg.get("model_path", "models/model.pkl")
    if os.path.exists(model_path):
        load_model(model_path)
    uvicorn.run(app, host="0.0.0.0", port=serve_cfg.get("port", 8000))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",   choices=["train","evaluate","serve"], default="train")
    p.add_argument("--config", default="configs/pipeline_config.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config) if os.path.exists(args.config) else {}

    if args.mode == "train":
        run_train(cfg)
    elif args.mode == "evaluate":
        run_evaluate(cfg)
    elif args.mode == "serve":
        run_serve(cfg)
