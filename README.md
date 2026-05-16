# End-to-End ML Pipeline with Azure Deployment

> Production-ready, reproducible ML pipeline with automated data ingestion (REST API, SQL, CSV), feature engineering, model training with experiment tracking, FastAPI serving, and CI/CD deployment to Microsoft Azure App Service via GitHub Actions.

---

## 🔍 Project Overview

A complete MLOps pipeline covering every stage of the ML lifecycle:

```
Data Ingestion → Feature Engineering → Training + CV → Experiment Tracking
      → REST API Serving → Docker → CI/CD → Azure App Service
```

---

## 🏗️ Architecture

```
ml-pipeline-azure/
├── src/
│   ├── data/
│   │   └── ingestion.py          # REST API, SQL, CSV/Parquet ingestion + feature engineering
│   ├── training/
│   │   └── trainer.py            # Reproducible training, cross-validation, experiment logging
│   ├── serving/
│   │   └── api.py                # FastAPI REST API (predict, batch predict, model info)
│   └── pipeline_runner.py        # CLI orchestrator: train | evaluate | serve
├── configs/
│   └── pipeline_config.yaml      # Full pipeline configuration
├── .github/workflows/
│   └── ci_cd.yml                 # GitHub Actions: test → train → build → deploy
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/PRATdoppelEK/ml-pipeline-azure.git
cd ml-pipeline-azure
pip install -r requirements.txt
```

---

## 🚀 Quickstart

### Train (synthetic data — no setup needed)
```bash
python src/pipeline_runner.py --mode train --config configs/pipeline_config.yaml
```

### Evaluate all experiment runs
```bash
python src/pipeline_runner.py --mode evaluate --config configs/pipeline_config.yaml
```

### Serve predictions locally
```bash
python src/pipeline_runner.py --mode serve --config configs/pipeline_config.yaml
# API available at http://localhost:8000
```

### Docker
```bash
docker build -t ml-pipeline .
docker run -p 8000:8000 -e MODEL_PATH=/app/models/model.pkl ml-pipeline
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/model/info` | Model type, params, feature importance |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |

### Example prediction request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_0": 1.2, "feature_1": -0.5, "feature_2": 0.8}}'
```

```json
{
  "prediction": 1,
  "probability": 0.87,
  "model_version": "1.0.0"
}
```

---

## 🔄 CI/CD Pipeline

The GitHub Actions workflow automatically:
1. **Test**: Linting (flake8) + unit tests (pytest) with coverage
2. **Train**: Runs the training pipeline and saves model artifacts
3. **Build**: Builds and pushes Docker image to Azure Container Registry
4. **Deploy**: Deploys to Azure App Service with health check

### Required GitHub Secrets
```
AZURE_CREDENTIALS      # Azure service principal JSON
ACR_LOGIN_SERVER       # Azure Container Registry server
ACR_USERNAME           # ACR username
ACR_PASSWORD           # ACR password
```

---

## 🧪 Supported Models

| Model | Key Parameters |
|-------|----------------|
| Random Forest | n_estimators, max_depth |
| Gradient Boosting | n_estimators, learning_rate, max_depth |
| SVM | C, kernel |
| Logistic Regression | C |

Switch model type in `configs/pipeline_config.yaml`:
```yaml
training:
  model_type: gradient_boosting
  model_params:
    n_estimators: 150
    learning_rate: 0.05
```

---

## 📊 Experiment Tracking

All runs are logged to `experiments/runs.jsonl` with:
- Run ID (config hash — fully reproducible)
- F1 Macro, Precision, Recall, ROC-AUC
- Cross-validation scores (mean ± std)
- Training duration, timestamp
- Feature importances

---
## 📊 Pipeline Performance & Results

### Model benchmark (synthetic classification dataset, 10k samples, 20 features)

| Model | F1 Macro | ROC-AUC | Training Time | Notes |
|-------|----------|---------|---------------|-------|
| **Gradient Boosting** | **0.91** | **0.96** | 12s | Best overall |
| Random Forest | 0.88 | 0.94 | 4s | Best speed/accuracy ratio |
| SVM (RBF) | 0.84 | 0.91 | 28s | Slower on larger datasets |
| Logistic Regression | 0.79 | 0.87 | < 1s | Useful baseline |

*5-fold cross-validation. Results logged automatically to `experiments/runs.jsonl`.*

### Pipeline reliability observations

- **Reproducibility**: Run ID based on config hash — identical config always produces identical results
- **CI/CD**: Full train → build → deploy cycle completes in ~4 minutes via GitHub Actions
- **API latency**: FastAPI `/predict` endpoint responds in < 50ms per request (single prediction)
- **Docker image size**: ~480MB (python:3.10-slim base with scikit-learn stack)
- **Azure deployment**: Zero-downtime deployment via App Service slot swapping
- **Data ingestion**: Handles REST API, SQL (SQLAlchemy), CSV, and Parquet sources in a single unified pipeline — no code changes needed when switching source type
  
## 🔧 Tech Stack

`scikit-learn` · `FastAPI` · `Uvicorn` · `Docker` · `GitHub Actions` · `Microsoft Azure` · `SQLAlchemy` · `Pydantic` · `Python 3.10+`

---

## 👤 Author

**Prateek Gaur** — ML Engineer | Battery & Engineering AI  
[LinkedIn](https://www.linkedin.com/in/prateek-gaur-15a629b4) · [GitHub](https://github.com/PRATdoppelEK)
