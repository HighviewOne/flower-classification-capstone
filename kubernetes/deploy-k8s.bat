@echo off
REM =============================================================================
REM Deploy Flower Classifier to Kubernetes (kind) - Windows Version
REM =============================================================================
REM Prerequisites:
REM   - Docker Desktop running
REM   - kind installed (winget install Kubernetes.kind)
REM   - kubectl installed (winget install Kubernetes.kubectl)
REM =============================================================================

echo ==============================================
echo FLOWER CLASSIFIER - KUBERNETES DEPLOYMENT
echo ==============================================

set CLUSTER_NAME=flower-cluster
set IMAGE_NAME=flower-classifier:latest

REM Step 1: Check prerequisites
echo.
echo [1/6] Checking prerequisites...

where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo X Docker not found. Please install Docker Desktop.
    exit /b 1
)

where kind >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo X kind not found. Install with: winget install Kubernetes.kind
    exit /b 1
)

where kubectl >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo X kubectl not found. Install with: winget install Kubernetes.kubectl
    exit /b 1
)

echo √ All prerequisites found

REM Step 2: Create kind cluster
echo.
echo [2/6] Setting up kind cluster...

kind get clusters 2>nul | findstr /C:"%CLUSTER_NAME%" >nul
if %ERRORLEVEL% equ 0 (
    echo √ Cluster '%CLUSTER_NAME%' already exists
) else (
    echo Creating cluster '%CLUSTER_NAME%'...
    kind create cluster --name %CLUSTER_NAME%
    if %ERRORLEVEL% neq 0 (
        echo X Failed to create cluster
        exit /b 1
    )
    echo √ Cluster created
)

REM Step 3: Check Docker image
echo.
echo [3/6] Checking Docker image...

docker images | findstr /C:"flower-classifier" >nul
if %ERRORLEVEL% equ 0 (
    echo √ Docker image exists
) else (
    echo Building Docker image...
    docker build -t flower-classifier -f docker/Dockerfile .
    if %ERRORLEVEL% neq 0 (
        echo X Failed to build image
        exit /b 1
    )
    echo √ Image built
)

REM Step 4: Load image into kind
echo.
echo [4/6] Loading image into kind cluster...
kind load docker-image %IMAGE_NAME% --name %CLUSTER_NAME%
if %ERRORLEVEL% neq 0 (
    echo X Failed to load image
    exit /b 1
)
echo √ Image loaded into cluster

REM Step 5: Deploy to Kubernetes
echo.
echo [5/6] Deploying to Kubernetes...

kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml

echo √ Deployment applied

REM Step 6: Wait for deployment
echo.
echo [6/6] Waiting for pods to be ready...

kubectl rollout status deployment/flower-classifier --timeout=120s

echo.
echo ==============================================
echo DEPLOYMENT COMPLETE!
echo ==============================================

REM Show pod status
echo.
echo Pod status:
kubectl get pods -l app=flower-classifier

echo.
echo ----------------------------------------------
echo TO ACCESS THE SERVICE:
echo ----------------------------------------------
echo.
echo Run this command to forward the port:
echo   kubectl port-forward service/flower-classifier 9696:80
echo.
echo Then test with:
echo   curl http://localhost:9696/health
echo.
echo Or run the test script:
echo   python tests/test_service.py
echo.
echo ----------------------------------------------

pause
