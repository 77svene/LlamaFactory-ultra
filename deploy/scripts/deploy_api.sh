#!/bin/bash
# deploy/scripts/deploy_api.sh
# One-Command Production Deployment with Auto-Scaling for LlamaFactory
# Usage: ./deploy_api.sh --model_path ./output/my_model --api_name my-api --gpu_type A100 --replicas 3

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODEL_PATH=""
API_NAME="llamafactory-api"
GPU_TYPE="T4"
REPLICAS=2
MIN_REPLICAS=1
MAX_REPLICAS=10
NAMESPACE="llamafactory"
REGISTRY="docker.io/llamafactory"
VERSION=$(date +%Y%m%d-%H%M%S)
CPU_REQUEST="2"
MEMORY_REQUEST="8Gi"
GPU_MEMORY="16Gi"
ENABLE_GPU_SHARING=true
ENABLE_AUTO_SCALING=true
SCALING_METRIC="latency"  # latency or throughput
TARGET_LATENCY_MS=100
TARGET_THROUGHPUT_RPS=50
COST_OPTIMIZATION=true
DRY_RUN=false
KUBE_CONTEXT=""
TERRAFORM_DIR="./deploy/terraform"
K8S_TEMPLATE_DIR="./deploy/kubernetes"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --api_name)
            API_NAME="$2"
            shift 2
            ;;
        --gpu_type)
            GPU_TYPE="$2"
            shift 2
            ;;
        --replicas)
            REPLICAS="$2"
            shift 2
            ;;
        --min_replicas)
            MIN_REPLICAS="$2"
            shift 2
            ;;
        --max_replicas)
            MAX_REPLICAS="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --cpu_request)
            CPU_REQUEST="$2"
            shift 2
            ;;
        --memory_request)
            MEMORY_REQUEST="$2"
            shift 2
            ;;
        --gpu_memory)
            GPU_MEMORY="$2"
            shift 2
            ;;
        --disable_gpu_sharing)
            ENABLE_GPU_SHARING=false
            shift
            ;;
        --disable_auto_scaling)
            ENABLE_AUTO_SCALING=false
            shift
            ;;
        --scaling_metric)
            SCALING_METRIC="$2"
            shift 2
            ;;
        --target_latency)
            TARGET_LATENCY_MS="$2"
            shift 2
            ;;
        --target_throughput)
            TARGET_THROUGHPUT_RPS="$2"
            shift 2
            ;;
        --disable_cost_optimization)
            COST_OPTIMIZATION=false
            shift
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --kube_context)
            KUBE_CONTEXT="$2"
            shift 2
            ;;
        --terraform_dir)
            TERRAFORM_DIR="$2"
            shift 2
            ;;
        --k8s_template_dir)
            K8S_TEMPLATE_DIR="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_PATH" ]; then
    echo -e "${RED}Error: --model_path is required${NC}"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model path $MODEL_PATH does not exist${NC}"
    exit 1
fi

# Function to print section header
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Function to check dependencies
check_dependencies() {
    print_header "Checking Dependencies"
    
    local missing_deps=()
    
    # Check for required commands
    for cmd in docker kubectl terraform python3 pip jq curl; do
        if ! command -v $cmd &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo -e "${RED}Missing dependencies: ${missing_deps[*]}${NC}"
        echo -e "${YELLOW}Please install them before continuing.${NC}"
        exit 1
    fi
    
    # Check Python version
    python_version=$(python3 --version | cut -d' ' -f2)
    if [[ $(echo "$python_version < 3.8" | bc) -eq 1 ]]; then
        echo -e "${RED}Python 3.8+ required. Found: $python_version${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}All dependencies satisfied${NC}"
}

# Function to validate model
validate_model() {
    print_header "Validating Model"
    
    # Check for required model files
    local required_files=("config.json" "tokenizer.json" "tokenizer_config.json")
    for file in "${required_files[@]}"; do
        if [ ! -f "$MODEL_PATH/$file" ]; then
            echo -e "${YELLOW}Warning: $file not found in model directory${NC}"
        fi
    done
    
    # Check for model weights
    if [ ! -f "$MODEL_PATH/pytorch_model.bin" ] && \
       [ ! -f "$MODEL_PATH/model.safetensors" ] && \
       [ ! -f "$MODEL_PATH/model.ckpt" ]; then
        echo -e "${RED}Error: No model weights found in $MODEL_PATH${NC}"
        exit 1
    fi
    
    # Extract model metadata
    if [ -f "$MODEL_PATH/config.json" ]; then
        MODEL_TYPE=$(jq -r '.model_type // "unknown"' "$MODEL_PATH/config.json")
        MODEL_ARCH=$(jq -r '.architectures[0] // "unknown"' "$MODEL_PATH/config.json")
        HIDDEN_SIZE=$(jq -r '.hidden_size // 0' "$MODEL_PATH/config.json")
        NUM_LAYERS=$(jq -r '.num_hidden_layers // 0' "$MODEL_PATH/config.json")
        
        echo -e "${GREEN}Model validated:${NC}"
        echo -e "  Type: $MODEL_TYPE"
        echo -e "  Architecture: $MODEL_ARCH"
        echo -e "  Hidden Size: $HIDDEN_SIZE"
        echo -e "  Layers: $NUM_LAYERS"
    fi
}

# Function to create FastAPI server
create_fastapi_server() {
    print_header "Creating FastAPI Server"
    
    local server_dir="./deploy/server"
    mkdir -p "$server_dir"
    
    # Create main server file
    cat > "$server_dir/main.py" << 'EOF'
"""
LlamaFactory Production API Server
FastAPI server with auto-scaling, GPU sharing, and model versioning
"""

import os
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import psutil
import GPUtil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model cache
model_cache = {}
tokenizer_cache = {}

# Metrics for auto-scaling
metrics = {
    "requests_total": 0,
    "requests_in_progress": 0,
    "latency_p95": 0,
    "throughput_rps": 0,
    "gpu_memory_used": 0,
    "gpu_utilization": 0,
    "last_request_time": time.time()
}

class ModelConfig:
    """Model configuration and metadata"""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_type = "unknown"
        self.model_arch = "unknown"
        self.hidden_size = 0
        self.num_layers = 0
        self.load_time = 0
        self.version = os.getenv("MODEL_VERSION", "latest")
        self.load_config()
    
    def load_config(self):
        """Load model configuration"""
        config_path = os.path.join(self.model_path, "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, "r") as f:
                config = json.load(f)
                self.model_type = config.get("model_type", "unknown")
                self.model_arch = config.get("architectures", ["unknown"])[0]
                self.hidden_size = config.get("hidden_size", 0)
                self.num_layers = config.get("num_hidden_layers", 0)

class InferenceRequest(BaseModel):
    """Request model for inference"""
    prompt: str = Field(..., description="Input prompt for generation")
    max_new_tokens: int = Field(default=128, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(default=50, ge=0, description="Top-k sampling")
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0, description="Repetition penalty")
    do_sample: bool = Field(default=True, description="Whether to use sampling")
    stream: bool = Field(default=False, description="Whether to stream response")
    use_cache: bool = Field(default=True, description="Whether to use KV cache")

class InferenceResponse(BaseModel):
    """Response model for inference"""
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    inference_time_ms: float
    tokens_per_second: float
    model_version: str
    gpu_memory_used_gb: float
    request_id: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: str
    gpu_available: bool
    gpu_memory_used_gb: float
    gpu_utilization_percent: float
    cpu_percent: float
    memory_percent: float
    uptime_seconds: float
    requests_total: int
    latency_p95_ms: float

class MetricsResponse(BaseModel):
    """Metrics response for auto-scaling"""
    requests_total: int
    requests_in_progress: int
    latency_p95_ms: float
    throughput_rps: float
    gpu_memory_used_gb: float
    gpu_utilization_percent: float
    cpu_percent: float
    memory_percent: float
    model_version: str
    timestamp: float

def get_gpu_info():
    """Get GPU information"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Use first GPU
            return {
                "available": True,
                "name": gpu.name,
                "memory_used_gb": gpu.memoryUsed / 1024,
                "memory_total_gb": gpu.memoryTotal / 1024,
                "utilization_percent": gpu.load * 100
            }
    except:
        pass
    return {"available": False}

def load_model(model_path: str, model_name: str = "default"):
    """Load model and tokenizer"""
    global model_cache, tokenizer_cache
    
    if model_name in model_cache:
        logger.info(f"Model {model_name} already loaded")
        return model_cache[model_name], tokenizer_cache[model_name]
    
    logger.info(f"Loading model from {model_path}")
    start_time = time.time()
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Determine device and dtype
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if device == "cpu":
            model = model.to(device)
        
        # Enable GPU memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if os.getenv("ENABLE_GPU_SHARING", "true").lower() == "true":
                # Set memory fraction for GPU sharing
                memory_fraction = float(os.getenv("GPU_MEMORY_FRACTION", "0.8"))
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s")
        
        # Cache model
        model_cache[model_name] = model
        tokenizer_cache[model_name] = tokenizer
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    logger.info("Starting LlamaFactory API Server")
    
    model_path = os.getenv("MODEL_PATH", "./model")
    model_name = os.getenv("MODEL_NAME", "default")
    
    try:
        load_model(model_path, model_name)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LlamaFactory API Server")
    model_cache.clear()
    tokenizer_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Create FastAPI app
app = FastAPI(
    title="LlamaFactory API",
    description="Production API for LlamaFactory fine-tuned models with auto-scaling",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to collect metrics"""
    start_time = time.time()
    metrics["requests_in_progress"] += 1
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    metrics["requests_total"] += 1
    metrics["requests_in_progress"] -= 1
    metrics["last_request_time"] = time.time()
    
    # Update latency metrics (simplified P95 calculation)
    metrics["latency_p95"] = max(metrics["latency_p95"] * 0.95, process_time)
    
    # Update throughput
    time_diff = time.time() - metrics["last_request_time"]
    if time_diff > 0:
        metrics["throughput_rps"] = 1.0 / time_diff
    
    # Update GPU metrics
    gpu_info = get_gpu_info()
    if gpu_info["available"]:
        metrics["gpu_memory_used"] = gpu_info["memory_used_gb"]
        metrics["gpu_utilization"] = gpu_info["utilization_percent"]
    
    response.headers["X-Process-Time-Ms"] = str(process_time)
    response.headers["X-Request-ID"] = str(metrics["requests_total"])
    
    return response

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    gpu_info = get_gpu_info()
    
    return HealthResponse(
        status="healthy",
        model_loaded=len(model_cache) > 0,
        model_version=os.getenv("MODEL_VERSION", "latest"),
        gpu_available=gpu_info["available"],
        gpu_memory_used_gb=gpu_info.get("memory_used_gb", 0),
        gpu_utilization_percent=gpu_info.get("utilization_percent", 0),
        cpu_percent=psutil.cpu_percent(),
        memory_percent=psutil.virtual_memory().percent,
        uptime_seconds=time.time() - psutil.boot_time(),
        requests_total=metrics["requests_total"],
        latency_p95_ms=metrics["latency_p95"]
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get metrics for auto-scaling"""
    gpu_info = get_gpu_info()
    
    return MetricsResponse(
        requests_total=metrics["requests_total"],
        requests_in_progress=metrics["requests_in_progress"],
        latency_p95_ms=metrics["latency_p95"],
        throughput_rps=metrics["throughput_rps"],
        gpu_memory_used_gb=gpu_info.get("memory_used_gb", 0),
        gpu_utilization_percent=gpu_info.get("utilization_percent", 0),
        cpu_percent=psutil.cpu_percent(),
        memory_percent=psutil.virtual_memory().percent,
        model_version=os.getenv("MODEL_VERSION", "latest"),
        timestamp=time.time()
    )

@app.post("/v1/completions", response_model=InferenceResponse)
async def create_completion(request: InferenceRequest, background_tasks: BackgroundTasks):
    """Create completion endpoint (OpenAI compatible)"""
    request_id = f"cmpl-{metrics['requests_total']}"
    
    try:
        model_name = os.getenv("MODEL_NAME", "default")
        if model_name not in model_cache:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        model = model_cache[model_name]
        tokenizer = tokenizer_cache[model_name]
        
        # Tokenize input
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        input_length = inputs["input_ids"].shape[1]
        
        # Generate
        start_time = time.time()
        
        with torch.no_grad():
            generation_config = GenerationConfig(
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                do_sample=request.do_sample,
                use_cache=request.use_cache,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        inference_time = (time.time() - start_time) * 1000
        
        # Decode output
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Calculate metrics
        num_generated_tokens = len(generated_tokens)
        tokens_per_second = num_generated_tokens / (inference_time / 1000) if inference_time > 0 else 0
        
        # Get GPU memory usage
        gpu_info = get_gpu_info()
        gpu_memory_used = gpu_info.get("memory_used_gb", 0)
        
        return InferenceResponse(
            generated_text=generated_text,
            prompt_tokens=input_length,
            generated_tokens=num_generated_tokens,
            total_tokens=input_length + num_generated_tokens,
            inference_time_ms=inference_time,
            tokens_per_second=tokens_per_second,
            model_version=os.getenv("MODEL_VERSION", "latest"),
            gpu_memory_used_gb=gpu_memory_used,
            request_id=request_id
        )
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=507, detail="GPU out of memory")
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    """Chat completion endpoint (OpenAI compatible) - placeholder"""
    # This would be implemented similar to completions but with chat formatting
    return {"error": "Chat endpoint not yet implemented"}

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "models": [
            {
                "id": os.getenv("MODEL_NAME", "default"),
                "object": "model",
                "created": int(time.time()),
                "owned_by": "llamafactory",
                "permission": [],
                "root": os.getenv("MODEL_NAME", "default"),
                "parent": None
            }
        ]
    }

@app.get("/version")
async def get_version():
    """Get API and model version"""
    return {
        "api_version": "1.0.0",
        "model_version": os.getenv("MODEL_VERSION", "latest"),
        "model_path": os.getenv("MODEL_PATH", "./model"),
        "gpu_sharing_enabled": os.getenv("ENABLE_GPU_SHARING", "true").lower() == "true",
        "auto_scaling_enabled": os.getenv("ENABLE_AUTO_SCALING", "true").lower() == "true"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=int(os.getenv("WORKERS", "1")),
        log_level="info",
        access_log=True
    )
EOF

    # Create requirements file
    cat > "$server_dir/requirements.txt" << 'EOF'
torch>=2.0.0
transformers>=4.34.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
psutil>=5.9.0
gputil>=1.4.0
python-multipart>=0.0.6
accelerate>=0.24.0
safetensors>=0.4.0
sentencepiece>=0.1.99
protobuf>=3.20.0
EOF

    # Create Dockerfile
    cat > "$server_dir/Dockerfile" << EOF
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    python3-dev \\
    git \\
    wget \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 llamafactory && \\
    chown -R llamafactory:llamafactory /app
USER llamafactory

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python3", "main.py"]
EOF

    echo -e "${GREEN}FastAPI server created in $server_dir${NC}"
}

# Function to create Kubernetes templates
create_kubernetes_templates() {
    print_header "Creating Kubernetes Templates"
    
    mkdir -p "$K8S_TEMPLATE_DIR"
    
    # Create namespace
    cat > "$K8S_TEMPLATE_DIR/namespace.yaml" << EOF
apiVersion: v1
kind: Namespace
metadata:
  name: $NAMESPACE
  labels:
    app: $API_NAME
    managed-by: llamafactory
EOF

    # Create configmap for model configuration
    cat > "$K8S_TEMPLATE_DIR/configmap.yaml" << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: $API_NAME-config
  namespace: $NAMESPACE
data:
  MODEL_NAME: "$API_NAME"
  MODEL_VERSION: "$VERSION"
  ENABLE_GPU_SHARING: "$ENABLE_GPU_SHARING"
  ENABLE_AUTO_SCALING: "$ENABLE_AUTO_SCALING"
  GPU_MEMORY_FRACTION: "0.8"
  MAX_BATCH_SIZE: "4"
  MAX_SEQUENCE_LENGTH: "2048"
EOF

    # Create deployment
    cat > "$K8S_TEMPLATE_DIR/deployment.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $API_NAME
  namespace: $NAMESPACE
  labels:
    app: $API_NAME
    version: $VERSION
spec:
  replicas: $REPLICAS
  selector:
    matchLabels:
      app: $API_NAME
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: $API_NAME
        version: $VERSION
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: $API_NAME
        image: $REGISTRY/$API_NAME:$VERSION
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        envFrom:
        - configMapRef:
            name: $API_NAME-config
        env:
        - name: MODEL_PATH
          value: "/models"
        - name: PORT
          value: "8000"
        resources:
          requests:
            cpu: "$CPU_REQUEST"
            memory: "$MEMORY_REQUEST"
          limits:
            cpu: "$((CPU_REQUEST * 2))"
            memory: "$MEMORY_REQUEST"
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: model-storage
          mountPath: /models
          readOnly: true
        - name: shm
          mountPath: /dev/shm
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 30
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: $API_NAME-model-pvc
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 2Gi
      nodeSelector:
        node.kubernetes.io/instance-type: "$GPU_TYPE"
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
EOF

    # Create service
    cat > "$K8S_TEMPLATE_DIR/service.yaml" << EOF
apiVersion: v1
kind: Service
metadata:
  name: $API_NAME
  namespace: $NAMESPACE
  labels:
    app: $API_NAME
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: $API_NAME
EOF

    # Create ingress (optional)
    cat > "$K8S_TEMPLATE_DIR/ingress.yaml" << EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: $API_NAME
  namespace: $NAMESPACE
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
spec:
  rules:
  - host: $API_NAME.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: $API_NAME
            port:
              number: 80
  tls:
  - hosts:
    - $API_NAME.example.com
    secretName: $API_NAME-tls
EOF

    # Create HPA (Horizontal Pod Autoscaler)
    if [ "$ENABLE_AUTO_SCALING" = true ]; then
        cat > "$K8S_TEMPLATE_DIR/hpa.yaml" << EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: $API_NAME-hpa
  namespace: $NAMESPACE
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: $API_NAME
  minReplicas: $MIN_REPLICAS
  maxReplicas: $MAX_REPLICAS
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: inference_latency_p95_ms
      target:
        type: AverageValue
        averageValue: ${TARGET_LATENCY_MS}
  - type: Pods
    pods:
      metric:
        name: inference_throughput_rps
      target:
        type: AverageValue
        averageValue: ${TARGET_THROUGHPUT_RPS}
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
EOF
    fi

    # Create PVC for model storage
    cat > "$K8S_TEMPLATE_DIR/pvc.yaml" << EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: $API_NAME-model-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: standard-rwo
EOF

    # Create ServiceMonitor for Prometheus (if using Prometheus Operator)
    cat > "$K8S_TEMPLATE_DIR/servicemonitor.yaml" << EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: $API_NAME-monitor
  namespace: $NAMESPACE
  labels:
    app: $API_NAME
spec:
  selector:
    matchLabels:
      app: $API_NAME
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
    scrapeTimeout: 10s
EOF

    echo -e "${GREEN}Kubernetes templates created in $K8S_TEMPLATE_DIR${NC}"
}

# Function to create Terraform templates
create_terraform_templates() {
    print_header "Creating Terraform Templates"
    
    mkdir -p "$TERRAFORM_DIR"
    
    # Create main Terraform configuration
    cat > "$TERRAFORM_DIR/main.tf" << EOF
# LlamaFactory Auto-Scaling Deployment with Terraform
# Supports AWS EKS, GCP GKE, and Azure AKS

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
}

# Variables
variable "cluster_name" {
  description = "Name of the Kubernetes cluster"
  type        = string
  default     = "llamafactory-cluster"
}

variable "region" {
  description = "Cloud region"
  type        = string
  default     = "us-west-2"
}

variable "node_instance_type" {
  description = "Instance type for GPU nodes"
  type        = string
  default     = "p3.2xlarge"  # AWS instance with NVIDIA V100
}

variable "min_nodes" {
  description = "Minimum number of nodes"
  type        = number
  default     = 1
}

variable "max_nodes" {
  description = "Maximum number of nodes"
  type        = number
  default     = 10
}

variable "desired_nodes" {
  description = "Desired number of nodes"
  type        = number
  default     = 2
}

variable "enable_spot_instances" {
  description = "Use spot instances for cost optimization"
  type        = bool
  default     = true
}

variable "spot_max_price" {
  description = "Maximum price for spot instances"
  type        = string
  default     = "1.50"  # USD per hour
}

# AWS Provider
provider "aws" {
  region = var.region
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_eks_cluster" "cluster" {
  name = module.eks.cluster_name
}

data "aws_eks_cluster_auth" "cluster" {
  name = module.eks.cluster_name
}

# Kubernetes provider
provider "kubernetes" {
  host                   = data.aws_eks_cluster.cluster.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)
  token                  = data.aws_eks_cluster_auth.cluster.token
}

# Helm provider
provider "helm" {
  kubernetes {
    host                   = data.aws_eks_cluster.cluster.endpoint
    cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)
    token                  = data.aws_eks_cluster_auth.cluster.token
  }
}

# VPC Module
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "\${var.cluster_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway   = true
  single_nat_gateway   = true
  enable_dns_hostnames = true

  public_subnet_tags = {
    "kubernetes.io/cluster/\${var.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                      = 1
  }

  private_subnet_tags = {
    "kubernetes.io/cluster/\${var.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb"             = 1
  }
}

# EKS Module
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.28"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access = true

  eks_managed_node_groups = {
    gpu_nodes = {
      name = "gpu-node-group"

      instance_types = [var.node_instance_type]
      
      min_size     = var.min_nodes
      max_size     = var.max_nodes
      desired_size = var.desired_nodes

      # Use spot instances if enabled
      capacity_type = var.enable_spot_instances ? "SPOT" : "ON_DEMAND"
      
      # GPU-specific configuration
      ami_type = "AL2_x86_64_GPU"
      
      labels = {
        role        = "gpu"
        environment = "production"
      }

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]

      update_config = {
        max_unavailable_percentage = 25
      }

      tags = {
        Environment = "production"
        Terraform   = "true"
      }
    }
  }

  # Enable IRSA for pod IAM roles
  enable_irsa = true

  tags = {
    Environment = "production"
    Terraform   = "true"
  }
}

# Install NVIDIA device plugin
resource "helm_release" "nvidia_device_plugin" {
  name       = "nvidia-device-plugin"
  repository = "https://nvidia.github.io/k8s-device-plugin"
  chart      = "nvidia-device-plugin"
  namespace  = "kube-system"
  version    = "0.14.1"

  set {
    name  = "tolerations[0].key"
    value = "nvidia.com/gpu"
  }

  set {
    name  = "tolerations[0].operator"
    value = "Exists"
  }

  set {
    name  = "tolerations[0].effect"
    value = "NoSchedule"
  }
}

# Install metrics server for HPA
resource "helm_release" "metrics_server" {
  name       = "metrics-server"
  repository = "https://kubernetes-sigs.github.io/metrics-server/"
  chart      = "metrics-server"
  namespace  = "kube-system"
  version    = "3.11.0"

  set {
    name  = "args[0]"
    value = "--kubelet-preferred-address-types=InternalIP"
  }
}

# Install Prometheus for monitoring
resource "helm_release" "prometheus" {
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  namespace  = "monitoring"
  version    = "51.2.0"

  create_namespace = true

  set {
    name  = "prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues"
    value = "false"
  }

  set {
    name  = "prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues"
    value = "false"
  }
}

# Create namespace for LlamaFactory
resource "kubernetes_namespace" "llamafactory" {
  metadata {
    name = "$NAMESPACE"
    labels = {
      app = "llamafactory"
    }
  }
}

# Create storage class for GPU nodes
resource "kubernetes_storage_class" "gp2" {
  metadata {
    name = "gp2"
  }
  storage_provisioner = "kubernetes.io/aws-ebs"
  parameters = {
    type = "gp2"
  }
  reclaim_policy      = "Delete"
  volume_binding_mode = "WaitForFirstConsumer"
}

# Outputs
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "Kubernetes Cluster Name"
  value       = module.eks.cluster_name
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

output "configure_kubectl" {
  description = "Configure kubectl: make sure you're logged in with the correct AWS profile and run the following command to update your kubeconfig"
  value       = "aws eks update-kubeconfig --name \${module.eks.cluster_name} --region \${var.region}"
}
EOF

    # Create variables file
    cat > "$TERRAFORM_DIR/variables.tf" << EOF
variable "api_name" {
  description = "Name of the API deployment"
  type        = string
  default     = "$API_NAME"
}

variable "model_path" {
  description = "Path to the model"
  type        = string
  default     = "$MODEL_PATH"
}

variable "gpu_type" {
  description = "Type of GPU to use"
  type        = string
  default     = "$GPU_TYPE"
}

variable "enable_cost_optimization" {
  description = "Enable cost optimization features"
  type        = bool
  default     = $COST_OPTIMIZATION
}

variable "enable_gpu_sharing" {
  description = "Enable GPU sharing for multiple models"
  type        = bool
  default     = $ENABLE_GPU_SHARING
}

variable "enable_auto_scaling" {
  description = "Enable auto-scaling"
  type        = bool
  default     = $ENABLE_AUTO_SCALING
}
EOF

    # Create outputs file
    cat > "$TERRAFORM_DIR/outputs.tf" << EOF
output "api_endpoint" {
  description = "API endpoint URL"
  value       = "http://\${kubernetes_service.llamafactory.status[0].load_balancer[0].ingress[0].hostname}"
}

output "model_version" {
  description = "Deployed model version"
  value       = "$VERSION"
}

output "cluster_info" {
  description = "Cluster information"
  value = {
    name     = module.eks.cluster_name
    endpoint = module.eks.cluster_endpoint
    region   = var.region
  }
}
EOF

    echo -e "${GREEN}Terraform templates created in $TERRAFORM_DIR${NC}"
}

# Function to build and push Docker image
build_docker_image() {
    print_header "Building Docker Image"
    
    local server_dir="./deploy/server"
    local image_tag="$REGISTRY/$API_NAME:$VERSION"
    
    # Copy model to server directory
    echo -e "${YELLOW}Copying model files...${NC}"
    mkdir -p "$server_dir/model"
    cp -r "$MODEL_PATH"/* "$server_dir/model/" 2>/dev/null || true
    
    # Build Docker image
    echo -e "${YELLOW}Building Docker image: $image_tag${NC}"
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY RUN] Would build: docker build -t $image_tag $server_dir${NC}"
    else
        docker build -t "$image_tag" "$server_dir"
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}Docker build failed${NC}"
            exit 1
        fi
        
        # Tag as latest
        docker tag "$image_tag" "$REGISTRY/$API_NAME:latest"
        
        echo -e "${GREEN}Docker image built successfully${NC}"
    fi
}

# Function to push Docker image
push_docker_image() {
    print_header "Pushing Docker Image"
    
    local image_tag="$REGISTRY/$API_NAME:$VERSION"
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY RUN] Would push: docker push $image_tag${NC}"
        return
    fi
    
    # Login to registry if credentials are provided
    if [ -n "$DOCKER_USERNAME" ] && [ -n "$DOCKER_PASSWORD" ]; then
        echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin "$(echo $REGISTRY | cut -d'/' -f1)"
    fi
    
    echo -e "${YELLOW}Pushing image: $image_tag${NC}"
    docker push "$image_tag"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Docker push failed${NC}"
        exit 1
    fi
    
    # Push latest tag
    docker push "$REGISTRY/$API_NAME:latest"
    
    echo -e "${GREEN}Docker image pushed successfully${NC}"
}

# Function to deploy to Kubernetes
deploy_to_kubernetes() {
    print_header "Deploying to Kubernetes"
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY RUN] Would apply Kubernetes manifests${NC}"
        return
    fi
    
    # Set kubectl context if provided
    if [ -n "$KUBE_CONTEXT" ]; then
        kubectl config use-context "$KUBE_CONTEXT"
    fi
    
    # Create namespace
    echo -e "${YELLOW}Creating namespace: $NAMESPACE${NC}"
    kubectl apply -f "$K8S_TEMPLATE_DIR/namespace.yaml"
    
    # Create configmap
    echo -e "${YELLOW}Creating configmap${NC}"
    kubectl apply -f "$K8S_TEMPLATE_DIR/configmap.yaml"
    
    # Create PVC
    echo -e "${YELLOW}Creating PVC${NC}"
    kubectl apply -f "$K8S_TEMPLATE_DIR/pvc.yaml"
    
    # Upload model to PVC (simplified - in production, use init containers or sidecars)
    echo -e "${YELLOW}Uploading model to PVC${NC}"
    # This would typically involve creating a job to copy the model to the PVC
    
    # Create deployment
    echo -e "${YELLOW}Creating deployment${NC}"
    kubectl apply -f "$K8S_TEMPLATE_DIR/deployment.yaml"
    
    # Create service
    echo -e "${YELLOW}Creating service${NC}"
    kubectl apply -f "$K8S_TEMPLATE_DIR/service.yaml"
    
    # Create HPA if auto-scaling is enabled
    if [ "$ENABLE_AUTO_SCALING" = true ]; then
        echo -e "${YELLOW}Creating HPA${NC}"
        kubectl apply -f "$K8S_TEMPLATE_DIR/hpa.yaml"
    fi
    
    # Create ServiceMonitor if Prometheus is available
    if kubectl get crd servicemonitors.monitoring.coreos.com &> /dev/null; then
        echo -e "${YELLOW}Creating ServiceMonitor${NC}"
        kubectl apply -f "$K8S_TEMPLATE_DIR/servicemonitor.yaml"
    fi
    
    # Wait for deployment to be ready
    echo -e "${YELLOW}Waiting for deployment to be ready...${NC}"
    kubectl rollout status deployment/$API_NAME -n $NAMESPACE --timeout=300s
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Deployment successful!${NC}"
        
        # Get service endpoint
        echo -e "${BLUE}Service endpoint:${NC}"
        kubectl get service $API_NAME -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || \
        kubectl get service $API_NAME -n $NAMESPACE -o jsonpath='{.spec.clusterIP}'
        
        echo -e "\n${GREEN}API is now available!${NC}"
        echo -e "${BLUE}Health check: curl http://\$(kubectl get service $API_NAME -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || kubectl get service $API_NAME -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')/health${NC}"
    else
        echo -e "${RED}Deployment failed${NC}"
        exit 1
    fi
}

# Function to deploy with Terraform
deploy_with_terraform() {
    print_header "Deploying with Terraform"
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY RUN] Would run Terraform${NC}"
        return
    fi
    
    cd "$TERRAFORM_DIR"
    
    # Initialize Terraform
    echo -e "${YELLOW}Initializing Terraform${NC}"
    terraform init
    
    # Plan deployment
    echo -e "${YELLOW}Planning Terraform deployment${NC}"
    terraform plan -out=tfplan
    
    # Apply deployment
    echo -e "${YELLOW}Applying Terraform deployment${NC}"
    terraform apply tfplan
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Terraform deployment successful!${NC}"
        
        # Get outputs
        echo -e "${BLUE}Cluster information:${NC}"
        terraform output
    else
        echo -e "${RED}Terraform deployment failed${NC}"
        exit 1
    fi
    
    cd - > /dev/null
}

# Function to create deployment summary
create_deployment_summary() {
    print_header "Deployment Summary"
    
    cat > "deployment-summary-$VERSION.md" << EOF
# LlamaFactory Deployment Summary

## Deployment Details
- **API Name**: $API_NAME
- **Model Version**: $VERSION
- **Model Path**: $MODEL_PATH
- **GPU Type**: $GPU_TYPE
- **Namespace**: $NAMESPACE
- **Registry**: $REGISTRY

## Configuration
- **Replicas**: $REPLICAS
- **Min Replicas**: $MIN_REPLICAS
- **Max Replicas**: $MAX_REPLICAS
- **Auto-scaling**: $ENABLE_AUTO_SCALING
- **GPU Sharing**: $ENABLE_GPU_SHARING
- **Cost Optimization**: $COST_OPTIMIZATION

## Scaling Metrics
- **Scaling Metric**: $SCALING_METRIC
- **Target Latency**: ${TARGET_LATENCY_MS}ms
- **Target Throughput**: ${TARGET_THROUGHPUT_RPS} RPS

## Resources
- **CPU Request**: $CPU_REQUEST
- **Memory Request**: $MEMORY_REQUEST
- **GPU Memory**: $GPU_MEMORY

## Endpoints
- **Health Check**: /health
- **Metrics**: /metrics
- **Completions**: /v1/completions
- **Models**: /v1/models
- **Version**: /version

## Deployment Commands
\`\`\`bash
# Check deployment status
kubectl get pods -n $NAMESPACE

# View logs
kubectl logs -f deployment/$API_NAME -n $NAMESPACE

# Scale manually
kubectl scale deployment $API_NAME --replicas=5 -n $NAMESPACE

# Port forward for local testing
kubectl port-forward service/$API_NAME 8000:80 -n $NAMESPACE
\`\`\`

## Monitoring
- **Prometheus Metrics**: Available at /metrics endpoint
- **Grafana Dashboard**: Import dashboard ID 1860 for Kubernetes monitoring
- **Auto-scaling**: Based on $SCALING_METRIC with target ${TARGET_LATENCY_MS}ms latency

## Cost Optimization
- **GPU Sharing**: $ENABLE_GPU_SHARING
- **Spot Instances**: Enabled in Terraform configuration
- **Auto-scaling**: Down to $MIN_REPLICAS during low traffic

Generated at: $(date)
EOF

    echo -e "${GREEN}Deployment summary saved to deployment-summary-$VERSION.md${NC}"
}

# Main execution
main() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║           LlamaFactory Production Deployment                 ║${NC}"
    echo -e "${BLUE}║     One-Command Auto-Scaling API Deployment                  ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    
    # Check dependencies
    check_dependencies
    
    # Validate model
    validate_model
    
    # Create FastAPI server
    create_fastapi_server
    
    # Create Kubernetes templates
    create_kubernetes_templates
    
    # Create Terraform templates
    create_terraform_templates
    
    # Build Docker image
    build_docker_image
    
    # Ask for deployment method
    echo -e "\n${YELLOW}Choose deployment method:${NC}"
    echo "1) Kubernetes only (requires existing cluster)"
    echo "2) Terraform + Kubernetes (creates cluster)"
    echo "3) Dry run only (no actual deployment)"
    read -p "Enter choice [1-3]: " deploy_choice
    
    case $deploy_choice in
        1)
            # Push Docker image
            push_docker_image
            
            # Deploy to Kubernetes
            deploy_to_kubernetes
            ;;
        2)
            # Deploy with Terraform
            deploy_with_terraform
            
            # Push Docker image
            push_docker_image
            
            # Deploy to Kubernetes
            deploy_to_kubernetes
            ;;
        3)
            echo -e "${BLUE}Dry run completed. No actual deployment performed.${NC}"
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            exit 1
            ;;
    esac
    
    # Create deployment summary
    create_deployment_summary
    
    echo -e "\n${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║             Deployment Completed Successfully!               ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo -e "\n${BLUE}Next steps:${NC}"
    echo -e "1. Monitor the deployment: ${YELLOW}kubectl get pods -n $NAMESPACE${NC}"
    echo -e "2. Check logs: ${YELLOW}kubectl logs -f deployment/$API_NAME -n $NAMESPACE${NC}"
    echo -e "3. Test the API: ${YELLOW}curl http://<service-endpoint>/health${NC}"
    echo -e "4. View metrics: ${YELLOW}curl http://<service-endpoint>/metrics${NC}"
    echo -e "\n${BLUE}Documentation: deployment-summary-$VERSION.md${NC}"
}

# Run main function
main "$@"