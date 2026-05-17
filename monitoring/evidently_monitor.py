"""
Data drift and model performance monitoring using Evidently AI.
Run periodically to detect when model needs retraining.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric


def generate_reference_data(n_samples: int = 1000, n_features: int = 20) -> pd.DataFrame:
    """Generate synthetic reference dataset (baseline training distribution)."""
    np.random.seed(42)
    data = {f"feature_{i}": np.random.normal(0, 1, n_samples) for i in range(n_features)}
    data["target"] = np.random.randint(0, 2, n_samples)
    return pd.DataFrame(data)


def generate_current_data(n_samples: int = 200, drift: bool = True) -> pd.DataFrame:
    """Generate current data — with drift injected in first 5 features."""
    np.random.seed(99)
    data = {}
    for i in range(20):
        if drift and i < 5:
            data[f"feature_{i}"] = np.random.normal(1.5, 1.5, n_samples)
        else:
            data[f"feature_{i}"] = np.random.normal(0, 1, n_samples)
    data["target"] = np.random.randint(0, 2, n_samples)
    return pd.DataFrame(data)


def run_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_dir: str = "monitoring/reports"
) -> dict:
    """Run Evidently data drift report and save HTML + JSON summary."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
    ])

    report.run(reference_data=reference_data, current_data=current_data)

    # Save HTML report
    html_path = f"{output_dir}/drift_report_{timestamp}.html"
    report.save_html(html_path)

    # Extract summary
    report_dict = report.as_dict()
    drift_metric = report_dict["metrics"][1]["result"]

    summary = {
        "timestamp": timestamp,
        "dataset_drift_detected": drift_metric.get("dataset_drift", False),
        "drifted_features": drift_metric.get("number_of_drifted_columns", 0),
        "total_features": drift_metric.get("number_of_columns", 0),
        "drift_share": drift_metric.get("share_of_drifted_columns", 0.0),
        "html_report": html_path,
        "status": "RETRAIN RECOMMENDED" if drift_metric.get("dataset_drift") else "OK"
    }

    summary_path = f"{output_dir}/drift_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("  DRIFT MONITORING REPORT")
    print(f"{'='*60}")
    print(f"  Timestamp        : {timestamp}")
    print(f"  Drift detected   : {'YES' if summary['dataset_drift_detected'] else 'NO'}")
    print(f"  Drifted features : {summary['drifted_features']} / {summary['total_features']}")
    print(f"  Drift share      : {summary['drift_share']:.1%}")
    print(f"  Status           : {summary['status']}")
    print(f"  HTML report      : {html_path}")
    print(f"{'='*60}\n")

    return summary


if __name__ == "__main__":
    print("Running Evidently monitoring pipeline...\n")

    reference_data = generate_reference_data(n_samples=1000)
    current_data   = generate_current_data(n_samples=200, drift=True)

    summary = run_drift_report(reference_data, current_data)

    if summary["dataset_drift_detected"]:
        print("ACTION REQUIRED: Significant data drift detected.")
        print("Recommendation : Retrain model with recent data.\n")
    else:
        print("Model monitoring OK — no significant drift detected.\n")
