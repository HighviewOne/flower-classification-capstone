#!/bin/bash
# =============================================================================
# Deploy Flower Classifier to Kubernetes (kind)
# =============================================================================
# This script deploys the flower classification service to a local Kubernetes
# cluster using kind (Kubernetes in Docker).
#
# Prerequisites:
#   - Docker Desktop running
#   - kind installed (winget install Kubernetes.kind)
#   - kubectl installed (winget install Kubernetes.kubectl)
#
# Usage:
#   ./deploy-k8s.sh
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "FLOWER CLASSIFIER - KUBERNETES DEPLOYMENT"
echo "=============================================="

# Configuration
CLUSTER_NAME="flower-classifier-cluster"
IMAGE_NAME="flower-classifier:latest"

# Step 1: Check prerequisites
echo ""
echo "[1/6] Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop."
    exit 1
fi

if ! command -v kind &> /dev/null; then
    echo "❌ kind not found. Install with: winget install Kubernetes.kind"
    exit 1
fi

if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Install with: winget install Kubernetes.kubectl"
    exit 1
fi

echo "✓ All prerequisites found"

# Step 2: Create kind cluster (if not exists)
echo ""
echo "[2/6] Setting up kind cluster..."

if kind get clusters 2>/dev/null | grep -q "$CLUSTER_NAME"; then
    echo "✓ Cluster '$CLUSTER_NAME' already exists"
else
    echo "Creating cluster '$CLUSTER_NAME'..."
    kind create cluster --name "$CLUSTER_NAME"
    echo "✓ Cluster created"
fi

# Step 3: Build Docker image (if needed)
echo ""
echo "[3/6] Checking Docker image..."

if docker images | grep -q "flower-classifier"; then
    echo "✓ Docker image exists"
else
    echo "Building Docker image..."
    docker build -t flower-classifier -f docker/Dockerfile .
    echo "✓ Image built"
fi

# Step 4: Load image into kind cluster
echo ""
echo "[4/6] Loading image into kind cluster..."
kind load docker-image "$IMAGE_NAME" --name "$CLUSTER_NAME"
echo "✓ Image loaded into cluster"

# Step 5: Deploy to Kubernetes
echo ""
echo "[5/6] Deploying to Kubernetes..."

kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml

echo "✓ Deployment applied"

# Step 6: Wait for deployment and get access info
echo ""
echo "[6/6] Waiting for pods to be ready..."

kubectl rollout status deployment/flower-classifier --timeout=120s

echo ""
echo "=============================================="
echo "DEPLOYMENT COMPLETE!"
echo "=============================================="

# Get pod status
echo ""
echo "Pod status:"
kubectl get pods -l app=flower-classifier

# Port forward instructions
echo ""
echo "----------------------------------------------"
echo "TO ACCESS THE SERVICE:"
echo "----------------------------------------------"
echo ""
echo "Run this command to forward the port:"
echo "  kubectl port-forward service/flower-classifier 9696:80"
echo ""
echo "Then test with:"
echo "  curl http://localhost:9696/health"
echo ""
echo "Or run the test script:"
echo "  python tests/test_service.py"
echo ""
echo "----------------------------------------------"
