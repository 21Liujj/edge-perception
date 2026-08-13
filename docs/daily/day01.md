
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

## Day 1 / Stage 1C - Pin NanoOWL Upstream

- Attempt time: 2026-08-13 09:58:46 CST
- Input upstream: `https://github.com/NVIDIA-AI-IOT/nanoowl`
- Intended output: Git submodule at `third_party/nanoowl` with the official upstream commit pinned by the parent repository.
- Result: FAIL - the Jetson resolved GitHub but both HTTPS clone attempts timed out connecting to `github.com:443`; no NanoOWL source, model weights, dependencies, or TensorRT engines were downloaded.
- Upstream commit, branch, commit date, license, and dependency metadata: unavailable because the official repository could not be cloned/read from this host.
- Parent repository commit: not created because Stage 1C did not complete.
- Environment integrity: no dependency installation, environment modification, upstream source edit, remote configuration, push, or later-stage work was performed.

## Day 1 / Stage 1C - Pin NanoOWL Upstream (retry)

- Attempt time: 2026-08-13 10:02 CST
- Input upstream: `https://github.com/NVIDIA-AI-IOT/nanoowl`
- Action: added NanoOWL as a git submodule at `third_party/nanoowl` via `git submodule add` (no mirror or alternate source used).
- Network: GitHub reachable (HTTP 200); clone succeeded.
- Upstream code: not modified. No dependencies installed. No model weights downloaded. No TensorRT engine built. No `.venv`/CUDA/TensorRT/PyTorch changes. No GitHub remote configured, no push, no remote repository created.
- Result: PASS.

### NanoOWL pinned metadata (from local Git and file reads)

| Item | Value |
|---|---|
| Upstream URL | `https://github.com/NVIDIA-AI-IOT/nanoowl` |
| Submodule path | `third_party/nanoowl` |
| HEAD commit hash | `fb553dee4c58f8c53ec2bf01f355a54648e67dbd` |
| Current branch | `main` |
| Latest commit date | 2025-02-05 17:01:22 -0800 |
| LICENSE | Apache License 2.0 (`LICENSE.md`) |
| `setup.py` | declares package `nanoowl`, version `0.0.0`; no `install_requires` |

### Main dependency names

Sources: README "Setup" section and `import`/`from` statements in the `nanoowl/` package. No `requirements.txt` is present in the repo.

- PyTorch (`torch`)
- Transformers (`transformers`)
- torch2trt (referenced in README Setup; external Jetson package)
- NVIDIA TensorRT (runtime, via torch2trt)
- torchvision
- OpenCV (`cv2`)
- numpy
- Pillow (`PIL`)
- matplotlib
- CLIP (`clip`)
- NanoSAM (optional, for instance segmentation example)

### Parent repository

- Parent commit created: `chore: pin NanoOWL upstream`.

## Day 1 / Stage 1D - NanoOWL Dependency Compatibility Audit

- Audit time: 2026-08-13 10:08 CST
- Scope: read-only. Probe interpreter: `/home/nvidia/jetson-deploy-lab/.venv/bin/python` (Python 3.10.12). No package installed/upgraded/downgraded. No NanoOWL source edit. No weights downloaded. No TensorRT engine built.
- NanoOWL pinned commit: `fb553dee4c58f8c53ec2bf01f355a54648e67dbd` (`main`).

### Dependency declaration sources (read-only)

- `setup.py`: declares package `nanoowl` v0.0.0; **no `install_requires`** — i.e. NanoOWL pins no dependency versions itself.
- No `requirements.txt`, `pyproject.toml`, or `setup.cfg`.
- README "Setup" instructions: (1) Install PyTorch; (2) Install torch2trt from `https://github.com/NVIDIA-AI-IOT/torch2trt`; (3) Install NVIDIA TensorRT; (4) `pip install transformers`; (5) optional NanoSAM.
- Reference `docker/23-01/Dockerfile` (base `nvcr.io/nvidia/pytorch:23.01-py3`): `pip install pillow --upgrade`; `pip install git+https://github.com/NVIDIA-AI-IOT/torch2trt.git`; `pip install transformers timm accelerate`; `pip install git+https://github.com/openai/CLIP.git`.
- Imports in the `nanoowl/` package: `argparse, clip, cv2, dataclasses, enum, json, matplotlib, numpy, os, PIL, subprocess, tempfile, time, torch, torchvision, transformers, typing`.
  - `clip` is used only in `nanoowl/clip_predictor.py` (tree-detection path), not the core `owl_predictor`.
  - `timm` / `accelerate` are not imported by the `nanoowl/` package itself but are pulled in by the reference Dockerfile as transformers companions.

### Audit results (.venv actual)

| Package | Import name | Status | Version | Note |
|---|---|---|---|---|
| torch | torch | AVAILABLE | 2.5.0a0+872d972e41.nv24.08 | Jetson nv build (system site-packages). Pre-built wheels only; no version pin from NanoOWL. |
| torchvision | torchvision | AVAILABLE | 0.20.0a0+afc54f7 | Matches torch 2.5 nv build. |
| tensorrt | tensorrt | AVAILABLE | 10.3.0 | System `/usr/lib/python3.10/dist-packages`. Newer than Dockerfile era; torch2trt must target this. |
| numpy | numpy | AVAILABLE | 1.23.5 | system site. |
| cv2 | cv2 | AVAILABLE | 4.10.0 | system site. |
| matplotlib | matplotlib | AVAILABLE | 3.10.3 | system site. |
| Pillow | PIL | VERSION_RISK | 9.0.1 | NanoOWL Dockerfile explicitly `pip install pillow --upgrade` to fix `UnidentifiedImageError`; .venv still on old 9.0.1 = the version the upstream upgrade targets. Image-decode failures possible at runtime. |
| transformers | transformers | MISSING | — | Required (imported by package + README). NanoOWL gives no version pin; OWL-ViT model support demands a sufficiently recent transformers (~4.x with OwlViT). Unspecified version = install-time risk. |
| torch2trt | torch2trt | MISSING | — | Required for engine build/inference (README step 2). Must be built from upstream git against the local torch 2.5.0a nv build and TensorRT 10.3 — Jetson-version-coupled, no wheel. |
| clip | clip | MISSING | — | Only needed by `clip_predictor.py` (tree detection). OpenAI CLIP, install via git per Dockerfile. Not required for the core `owl_predict` path. |
| timm | timm | MISSING | — | Not imported by `nanoowl/` package; Dockerfile companion to transformers. Risk only if selected transformers version needs it. |
| accelerate | accelerate | MISSING | — | Not imported by `nanoowl/` package; Dockerfile companion to transformers. Risk only if selected transformers version needs it. |

### Summary

- AVAILABLE (6): torch 2.5.0a0+872d972e41.nv24.08, torchvision 0.20.0a0+afc54f7, tensorrt 10.3.0, numpy 1.23.5, cv2 4.10.0, matplotlib 3.10.3
- MISSING (5): transformers, torch2trt, clip (tree path only), timm, accelerate
- VERSION_RISK (1): Pillow/PIL 9.0.1 (upstream Dockerfile upgrades it to fix UnidentifiedImageError)

### NanoOWL explicit version / install constraints

- NanoOWL **declares no pinned dependency versions** (`setup.py` has no `install_requires`; no requirements/pyproject files).
- Hard install-method constraints (README + Dockerfile), all unsatisfied-locally items must follow these:
  - torch2trt: install from git `https://github.com/NVIDIA-AI-IOT/torch2trt.git` (not a PyPI wheel) — must match local torch 2.5.0a nv build + TensorRT 10.3.
  - transformers: `pip install transformers` (version unpinned by NanoOWL; choose one with OWL-ViT support).
  - clip: `pip install git+https://github.com/openai/CLIP.git`.
  - pillow: upstream explicitly upgrades to avoid `UnidentifiedImageError` (current .venv 9.0.1 is the flagged-old line).
  - torch / torchvision: Jetson pre-built nv wheels only; do not replace with stock PyPI wheels.
- Reference container is `nvcr.io/nvidia/pytorch:23.01-py3` (much older stack) — treat as install-method reference only, not a version requirement.

### Environment integrity

- No package install/upgrade/downgrade, no source edit, no weights download, no TensorRT engine build, no remote/push performed. Only `docs/daily/day01.md` was appended and a local commit created.
