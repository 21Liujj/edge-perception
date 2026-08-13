
## Day 1 / Stage 1A - Environment and Stage 0 Audit


- Audit time: 2026-08-13 09:30:41 CST
- Scope: read-only environment and existing Stage 0 audit; no package installation, upgrade, deletion, NanoOWL setup, or Stage 1B work performed.
- Stage 0 project path: `/home/nvidia/jetson-deploy-lab`
- Project virtual environment: `/home/nvidia/jetson-deploy-lab/.venv`
- Python distinction: system `/usr/bin/python3` is Python 3.10.12; project `/home/nvidia/jetson-deploy-lab/.venv/bin/python` is Python 3.10.12 with `include-system-site-packages = true` and project-local packages taking precedence.

### Platform and software

| Item | Current state |
|---|---|
| L4T / JetPack | L4T R36.4.3 (package 36.4.3-20250107174145), JetPack 6.2 family; `nvidia-jetpack` meta-package is not installed |
| CUDA | Toolkit 12.6.68 (SDK manifest 12.6.11); `nvcc` is available through project CUDA PATH, not the default shell PATH |
| TensorRT | 10.3.0.30 packages; Python 10.3.0 |
| System Python | 3.10.12; PyTorch 2.5.0a0+872d972e41.nv24.8; ONNX not installed in system site; ONNX Runtime GPU 1.20.0; protobuf 3.12.4 |
| Project .venv | Python 3.10.12; PyTorch 2.5.0a0+872d972e41.nv24.8; ONNX 1.17.0; ONNX Runtime GPU 1.20.0; protobuf 4.25.8; TensorRT 10.3.0 |
| RAM | 7.4 GiB total, 1.2 GiB used, 6.0 GiB available; swap 3.7 GiB total, 0 used |
| Root/project disk | 233 GiB total, 38 GiB used, 186 GiB available, 17% used; inodes 3% used |

### Stage 0 integrity

- PRESENT: `/home/nvidia/jetson-deploy-lab/.venv/bin/python`
- PRESENT: `/home/nvidia/jetson-deploy-lab/scripts/smoke_onnx_trt.sh`
- PRESENT: `/home/nvidia/jetson-deploy-lab/scripts/inventory.sh`
- PRESENT: `/home/nvidia/jetson-deploy-lab/docs/environment.md`
- PRESENT: `/home/nvidia/jetson-deploy-lab/docs/project-plan.md`
- Git status at Stage 1A audit time: unavailable because the project had not yet been initialized as a Git repository.
- Audit result: PASS - Stage 0 files and environment were present; note: the project had not previously been initialized with Git.

## Day 1 / Stage 1B - Git Baseline

- Repository initialized at `/home/nvidia/jetson-deploy-lab` without moving or deleting Stage 0 files.
- Added `.gitignore` rules for `.venv`, Python caches, TensorRT engines, model weights, model download caches, build products, and temporary files.
- `docs/`, `scripts/`, and Stage 0 evidence/configuration remain eligible for version control.
- Git identity: repo-local `user.name` and `user.email` were configured from values supplied by the project owner; no global Git configuration was changed.
- Initial commit: PASS - `e97b98276a71283836cfd86407e96e7f03d4c07b` (`baseline: validated Stage 0 ONNX-TensorRT pipeline`).
- Stage 1B status: PASS; repository baseline completed and protected environment/model artifacts remain ignored.
- No NanoOWL installation or Stage 1C work was performed, and the validated software environment was not changed.
