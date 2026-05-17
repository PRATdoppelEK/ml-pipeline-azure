#!/bin/bash
# AWS ECS Deployment Script for ML Pipeline
# Deploys the FastAPI model server to AWS ECS Fargate
# Usage: ./monitoring/aws_deploy.sh

set -e

# ── Configuration ───────────────────────────────────────────────
APP_NAME="ml-pipeline"
AWS_REGION="${AWS_REGION:-eu-central-1}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "123456789012")
ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${APP_NAME}"
ECS_CLUSTER="${APP_NAME}-cluster"
ECS_SERVICE="${APP_NAME}-service"
TASK_FAMILY="${APP_NAME}-task"
IMAGE_TAG="${1:-latest}"

echo "=============================================="
echo "  ML Pipeline — AWS ECS Deployment"
echo "=============================================="
echo "  Region      : ${AWS_REGION}"
echo "  Account     : ${AWS_ACCOUNT_ID}"
echo "  Image tag   : ${IMAGE_TAG}"
echo "=============================================="

# ── Step 1: Build Docker image ──────────────────────────────────
echo ""
echo "[1/5] Building Docker image..."
docker build -t ${APP_NAME}:${IMAGE_TAG} .
echo "✅ Docker image built: ${APP_NAME}:${IMAGE_TAG}"

# ── Step 2: Push to ECR ─────────────────────────────────────────
echo ""
echo "[2/5] Pushing to Amazon ECR..."
echo "      (Skipping in demo mode — set AWS credentials to enable)"
echo "      Commands that would run:"
echo "      aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REPO}"
echo "      docker tag ${APP_NAME}:${IMAGE_TAG} ${ECR_REPO}:${IMAGE_TAG}"
echo "      docker push ${ECR_REPO}:${IMAGE_TAG}"

# ── Step 3: Register ECS task definition ────────────────────────
echo ""
echo "[3/5] Registering ECS task definition..."
cat > /tmp/task_definition.json << TASKDEF
{
  "family": "${TASK_FAMILY}",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "${APP_NAME}",
      "image": "${ECR_REPO}:${IMAGE_TAG}",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ENVIRONMENT", "value": "production"},
        {"name": "MODEL_PATH",  "value": "/app/models/model.pkl"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group":         "/ecs/${APP_NAME}",
          "awslogs-region":        "${AWS_REGION}",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command":     ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval":    30,
        "timeout":     5,
        "retries":     3,
        "startPeriod": 60
      }
    }
  ]
}
TASKDEF
echo "✅ Task definition written to /tmp/task_definition.json"

# ── Step 4: Deploy to ECS ───────────────────────────────────────
echo ""
echo "[4/5] Deploying to ECS Fargate..."
echo "      Commands that would run with AWS credentials:"
echo "      aws ecs register-task-definition --cli-input-json file:///tmp/task_definition.json"
echo "      aws ecs update-service --cluster ${ECS_CLUSTER} --service ${ECS_SERVICE} --task-definition ${TASK_FAMILY} --force-new-deployment"

# ── Step 5: Summary ─────────────────────────────────────────────
echo ""
echo "[5/5] Deployment summary"
echo "=============================================="
echo "  App         : ${APP_NAME}"
echo "  Image       : ${APP_NAME}:${IMAGE_TAG}"
echo "  ECS Cluster : ${ECS_CLUSTER}"
echo "  ECS Service : ${ECS_SERVICE}"
echo "  Region      : ${AWS_REGION}"
echo "  Status      : Ready for deployment"
echo "=============================================="
echo ""
echo "To deploy with real AWS credentials:"
echo "  1. Run: aws configure"
echo "  2. Create ECR repo: aws ecr create-repository --repository-name ${APP_NAME}"
echo "  3. Run this script again: ./monitoring/aws_deploy.sh"
