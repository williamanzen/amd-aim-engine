# Comparison of NVIDIA NIM Operator and AIM Engine

## Purpose
- **k8s-nim-operator**: Kubernetes Operator for deploying, managing, and scaling NVIDIA NIM microservices.
- **aim-engine**: AMD Inference Microservice for deploying AI models on AMD GPUs with optimized configurations, caching, and smart loading.

## Language & Framework
- **k8s-nim-operator**: Go with Kubebuilder and controller-runtime.
- **aim-engine**: Python 3.8+ packaged with setuptools; uses YAML/JSON for configuration.

## Entry Points / APIs
- **k8s-nim-operator**:
  - CustomResourceDefinitions (CRDs): `NIMCache`, `NIMService`, `NIMPipeline`, `NemoCustomizer`, `NemoEvaluator`, `NemoGuardrail`.
  - Entrypoint: `cmd/main.go`.
- **aim-engine**:
  - Console scripts (`setup.py` entry_points): `aim-generate`, `aim-recipe-selector`, `aim-config-generator`, `aim-cache-manager`.

## Prerequisites
- **k8s-nim-operator**: Kubernetes v1.28+, NVIDIA GPUs supporting NIM microservices.
- **aim-engine**: AMD GPU with ROCm support; Docker; Python dependencies (`requests`, `PyYAML`, `jsonschema`); 16GB+ RAM recommended.

## Build & Deployment
- **k8s-nim-operator**:
  - Makefile targets for build, push, install, deploy, uninstall (in `deployments/container/Makefile`).
  - Generates installation manifests via Kustomize (`make build-installer`).
- **aim-engine**:
  - Shell scripts in `scripts/` for building Docker images and cleanup.
  - Kubernetes deployment and cleanup scripts in `k8s/scripts/`.
  - Python packaging via `setup.py`.

## Directory Structure
- **k8s-nim-operator**:
  - `api/`, `cmd/`, `config/`, `internal/`, `deployments/`, `bundle/`, `hack/`, `test/`, `tools/`, `vendor/`.
- **aim-engine**:
  - `src/aim_engine/`, `config/`, `scripts/`, `k8s/`, `examples/`, `tests/`, `docs/`.

## Testing
- **k8s-nim-operator**: End-to-end tests under `test/e2e/`; utility scripts in `hack/`.
- **aim-engine**: Unit and integration tests in `tests/`; example scripts under `examples/`.

## Licensing & Contribution
- **k8s-nim-operator**: Apache-2.0 licensed; contribution guidelines in `CONTRIBUTING.md`; `CODEOWNERS` defined.
- **aim-engine**: MIT licensed; no explicit contribution guidelines.

## Documentation
- **k8s-nim-operator**: README.md and official NVIDIA-hosted docs.
- **aim-engine**: README.md; detailed guides under `docs/`; example instructions in `examples/README.md`.

## Usage & Deployment Approach

### AIM Engine (Single Container) Restrictions:
- **GPU Scalability**: Limited to single-node deployments (1-8 GPUs max per container)
- **Multi-Model Serving**: Each model requires a separate container instance
- **Resource Isolation**: No fine-grained resource management between models
- **Auto-scaling**: Basic container-level scaling only
- **Distributed Inference**: No native support for multi-node tensor/pipeline parallelism
- **Load Balancing**: Must be handled externally (ingress, service mesh)
- **Configuration Management**: Recipe-based but limited to predefined GPU counts (1, 2, 4, 8)

### NIM Operator (Kubernetes-Native) Advantages:
- **Multi-Node Scaling**: Native support for distributed inference across multiple nodes
- **Tensor & Pipeline Parallelism**: Built-in support for tensor parallelism (`tensorParallelSize`) and pipeline parallelism (`pipelineParallelSize`) 
- **Advanced Auto-scaling**: Kubernetes HPA/VPA integration with custom metrics
- **Resource Management**: Fine-grained GPU allocation, DRA (Dynamic Resource Allocation) support
- **Multi-Model Orchestration**: Multiple NIM services with centralized management
- **Service Mesh Integration**: Native Kubernetes networking and service discovery
- **Rolling Updates**: Zero-downtime deployments with Kubernetes deployment strategies
- **Observability**: Built-in monitoring, logging, and metrics collection
- **High Availability**: Pod disruption budgets, node affinity, anti-affinity rules
