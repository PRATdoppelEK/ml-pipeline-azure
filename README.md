# End-to-End ML Pipeline with Azure Deployment

> Production-ready, reproducible ML pipeline with automated data ingestion (REST API, SQL, CSV), feature engineering, model training with experiment tracking, FastAPI serving, and CI/CD deployment to Microsoft Azure App Service via GitHub Actions.

---

## Pipeline overview

```
Data Ingestion → Feature Engineering → Training + CV → Experiment Tracking
      → REST API Serving → Docker → CI/CD → Azure App Service
```

---

## Architecture

```
ml-pipeline-azure/
├── src/
│   ├── data/
│   │   └── ingestion.py          # REST API, SQL, CSV ingestion + feature engineering
│   ├── training/
│   │   └── trainer.py            # Reproducible training, cross-validation, experiment logging
│   ├── serving/
│   │   └── api.py                # FastAPI REST endpoint (predict, batch, model info)
│   └── pipeline_runner.py        # CLI orchestrator: train | evaluate | serve
├── configs/
│   └── pipeline_config.yaml      # Full pipeline configuration
├── .github/workflows/
│   └── ci_cd.yml                 # GitHub Actions: test → train → Docker build → Azure deploy
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/PRATdoppelEK/ml-pipeline-azure.git
cd ml-pipeline-azure
pip install -r requirements.txt
```

---

## Quickstart

### Train (synthetic data — no setup needed)
```bash
python src/pipeline_runner.py --mode train --config configs/pipeline_config.yaml
```

### Evaluate
```bash
python src/pipeline_runner.py --mode evaluate --config configs/pipeline_config.yaml
```

### Serve predictions locally
```bash
python src/pipeline_runner.py --mode serve --config configs/pipeline_config.yaml
# API docs at http://localhost:8000/docs
```

### Docker
```bash
docker build -t ml-pipeline .
docker run -p 8000:8000 ml-pipeline
```

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/model/info` | Model type, version, feature importances |
| POST | `/predict` | Single prediction with probability |
| POST | `/predict/batch` | Batch predictions |

---

## CI/CD pipeline (GitHub Actions)

1. **Test** — flake8 linting + pytest with coverage
2. **Train** — runs training pipeline, saves model artifact
3. **Build** — Docker image pushed to Azure Container Registry
4. **Deploy** — deploys to Azure App Service with health check

---

## Supported models

`Random Forest` · `Gradient Boosting` · `SVM` · `Logistic Regression`

Switch model type in `configs/pipeline_config.yaml`.

---

## Tech stack

`scikit-learn` · `FastAPI` · `Uvicorn` · `Docker` · `GitHub Actions` · `Microsoft Azure` · `SQLAlchemy` · `Pydantic` · `Python 3.10+`

---

## Author

**Prateek Gaur** — ML Engineer | Battery & Engineering AI
[LinkedIn](https://www.linkedin.com/in/prateek-gaur-15a629b4) · [GitHub](https://github.com/PRATdoppelEK) · prateekgaur@gmx.de
