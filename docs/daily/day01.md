
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

## Day 1 / Stage 1E - NanoOWL Dependency Install Rehearsal (dry-run)

- Rehearsal time: 2026-08-13 10:15 CST
- Probe interpreter: `/home/nvidia/jetson-deploy-lab/.venv/bin/python` (pip 26.2.1). `.venv` has `include-system-site-packages = true`; torch/torchvision/numpy live in `/usr/local/lib/python3.10/dist-packages`, PIL in `/usr/lib/python3/dist-packages`, tensorrt in `/usr/lib/python3.10/dist-packages` — all system-wide, shared by the whole Jetson image.
- Method: `pip install --dry-run` only. **No package installed, upgraded, downgraded, or replaced.** `.venv` untouched. No model download. No engine build.
- Based on Stage 1D verdict: MISSING transformers, torch2trt, clip (tree-only), timm, accelerate; VERSION_RISK PIL 9.0.1.

### Dry-run results

| Command (dry-run) | Would install (new) | Touches torch / torchvision / numpy / tensorrt? |
|---|---|---|
| `pip install transformers` | transformers-5.15.0, tokenizers-0.22.2, huggingface_hub-1.27.0, safetensors-0.8.0, hf-xet-1.6.0, regex, typer, click, httpx, httpcore, h11, anyio, etc. | **No.** numpy stays 1.23.5 (already satisfied); torch/torchvision/tensorrt not in install set. |
| `pip install transformers timm accelerate` | above + timm-1.0.28, accelerate-1.14.0 | **No.** timm/accelerate see torch 2.5.0a0 nv, torchvision 0.20.0a0, numpy 1.23.5, PIL 9.0.1 — all "already satisfied". No change. |
| `pip install transformers Pillow --upgrade` | above + pillow-12.3.0 | **No** to torch stack; **Yes to Pillow** (system PIL 9.0.1 → 12.3.0, installed into user/venv). |
| `pip install "transformers==4.46.3"` | transformers-4.46.3, tokenizers-0.20.3, huggingface_hub-0.36.2, safetensors, regex, hf-xet | **No.** Same stack untouched. Fewer companion packages than the unpinned v5 line. |

#### Key finding on transformers version

- Unpinned `pip install transformers` pulls **transformers 5.15.0** (a major v5 jump) with tokenizers 0.22 + huggingface_hub 1.27. This is much newer than the NanoOWL reference Dockerfile era (~4.x). OWL-ViT model coverage in v5 is unverified and the API surface drifts across majors.
- Recommendation: **pin transformers to a known OWL-ViT-supporting 4.x** (e.g. `transformers==4.46.3`) rather than accept the resolvers default v5. Dry-run confirms 4.46.3 also leaves the torch stack untouched.

## Day 1 / Stage 1E - NanoOWL Dependency Install Rehearsal (dry-run)

- Rehearsal time: 2026-08-13 10:15 CST
- Probe interpreter: `/home/nvidia/jetson-deploy-lab/.venv/bin/python` (pip 26.2.1). `.venv` has `include-system-site-packages = true`; torch/torchvision/numpy live in `/usr/local/lib/python3.10/dist-packages`, PIL in `/usr/lib/python3/dist-packages`, tensorrt in `/usr/lib/python3.10/dist-packages` — all system-wide, shared by the whole Jetson image.
- Method: `pip install --dry-run` only. **No package installed, upgraded, downgraded, or replaced.** `.venv` untouched. No model download. No engine build.
- Based on Stage 1D verdict: MISSING transformers, torch2trt, clip (tree-only), timm, accelerate; VERSION_RISK PIL 9.0.1.

### Dry-run results

| Command (dry-run) | Would install (new) | Touches torch / torchvision / numpy / tensorrt? |
|---|---|---|
| `pip install transformers` | transformers-5.15.0, tokenizers-0.22.2, huggingface_hub-1.27.0, safetensors-0.8.0, hf-xet-1.6.0, regex, typer, click, httpx, httpcore, h11, anyio, annotated-doc, shellingham, exceptiongroup | **No.** numpy stays 1.23.5 (already satisfied); torch/torchvision/tensorrt not in install set. |
| `pip install transformers timm accelerate` | above + timm-1.0.28, accelerate-1.14.0 | **No.** timm/accelerate see torch 2.5.0a0 nv, torchvision 0.20.0a0, numpy 1.23.5, PIL 9.0.1 — all "already satisfied". No change. |
| `pip install transformers Pillow --upgrade` | above + pillow-12.3.0 | **No** to torch stack; **Yes to Pillow** (system PIL 9.0.1 → 12.3.0, installed into user/venv overlay). |
| `pip install "transformers==4.46.3"` | transformers-4.46.3, tokenizers-0.20.3, huggingface_hub-0.36.2, safetensors, regex, hf-xet | **No.** Same stack untouched. Fewer companion packages than the unpinned v5 line. |

#### Key finding on transformers version

- Unpinned `pip install transformers` pulls **transformers 5.15.0** (a major v5 jump) with tokenizers 0.22 + huggingface_hub 1.27. This is much newer than the NanoOWL reference Dockerfile era (~4.x). OWL-ViT model coverage in v5 is unverified and the API surface drifts across majors.
- Recommendation: **pin transformers to a known OWL-ViT-supporting 4.x** (e.g. `transformers==4.46.3`) rather than accept the resolver's default v5. Dry-run confirms 4.46.3 also leaves the torch stack untouched.

#### Pillow upgrade

- NanoOWL Dockerfile runs `pip install pillow --upgrade` to fix `UnidentifiedImageError`. Dry-run shows it would replace system PIL 9.0.1 with Pillow 12.3.0.
- Because `.venv` has `include-system-site-packages=true` and PIL sits in system dirs, an upgrade installs the newer Pillow into the venv/user overlay **without removing the system file**; import precedence then resolves to the newer copy. Net effect on the shared Jetson image is low (system package stays) but the venv's effective Pillow changes 9.0.1 → 12.3.0. Treat as: optional, isolate to venv, verify image decode after.

### torch2trt analysis (NOT installed)

- Reference install: `pip install git+https://github.com/NVIDIA-AI-IOT/torch2trt.git`.
- Dry-run (`--dry-run git+...torch2trt.git`) cloned the repo but produced **no "Would install" line** — its `setup.py` builds a C++/CUDA extension against the locally-linked torch + TensorRT and resolves deps at build time rather than via declared `install_requires`. Build-time, not declarative.
- Risk conclusion (from build nature + locality; a standalone repo re-clone for setup.py inspection timed out on this pass, so deps inferred from dry-run behavior):
  1. **Will not declaratively change torch / torchvision / numpy / tensorrt** — torch2trt depends on the existing torch+CUDA+TensorRT it is compiled against; it does not list them as installable requirements.
  2. **Build-time coupling risk is real but confined**: torch2trt compiles against the present Jetson `torch 2.5.0a0+872d972e41.nv24.08` + `tensorrt 10.3`. Must be built on this exact image; a stock PyPI torch would be incompatible but is **not** being pulled here.
  3. It may emit `torch.utils.cpp_extension` build artifacts / `~/.cache/torch_extensions` — environment cache only, not a torch version change.
  4. Recommendation: build torch2trt only when reaching the engine-build stage; keep it out of the Python-only minimal set now. Pre-flight a real install with `--dry-run` again right before, and confirm `torch.cuda.is_available()` and the TensorRT Python binding version first.

### Would pip modify torch / torchvision / numpy / tensorrt?

- **No** for every dry-run variant above (transformers alone, transformers+timm+accelerate, pinned 4.46.3, +Pillow upgrade). torch 2.5.0a0 nv, torchvision 0.20.0a0, numpy 1.23.5, tensorrt 10.3 all report "already satisfied" or are absent from install sets.
- **Pillow** is the only existing package a dry-run plans to change (9.0.1 → 12.3.0), and only when explicitly upgraded.

### Minimum install set

**NanoOWL main-line required (install in this order):**
1. `pip install "transformers==4.46.3"` — pin to OWL-ViT-supporting 4.x (avoid v5 default).
2. (optional, only if chosen transformers demands it) `pip install timm accelerate` — dry-run shows both currently pull in cleanly; defer unless transformers 4.46.3 import errors.
3. `pip install Pillow --upgrade` — apply the upstream UnidentifiedImageError fix into the venv overlay; verify `PIL.__version__` after.

**Tree Detection optional (NOT main-line, defer):**
- `pip install git+https://github.com/openai/CLIP.git` — only for `nanoowl/clip_predictor.py`.

**Not needed now:**
- timm / accelerate unless a transformers import fails.
- torch2trt — defer to engine-build stage (Jetson build-time component, not a pure Python dep). Engine build later will require it.

### Integrity

- `.venv` not modified; no install/upgrade/downgrade executed. No model weights. No engine. No remote/push. Only `docs/daily/day01.md` appended + one local commit.

## Day 1 / Stage 1F - Install Pinned Transformers + Regression

- Action time: 2026-08-13 10:30 CST
- Installed into project `.venv` (`/home/nvidia/jetson-deploy-lab/.venv/bin/python`): `transformers==4.46.3` only.
- Pulled companions (resolver-chosen, all new): tokenizers-0.20.3, safetensors-0.8.0, huggingface_hub-0.36.2, regex-2026.7.19, hf-xet-1.6.0.
- NOT installed / NOT upgraded: timm, accelerate, torch2trt, clip. Pillow NOT upgraded (still 9.0.1). No model weights downloaded. No NanoOWL engine built. No torch2trt build.
- Install result: `Successfully installed transformers-4.46.3` (exit 0).

### Post-install version snapshot (vs. Stage 1D baseline)

| Package | Baseline | Post-install | Changed? |
|---|---|---|---|
| transformers | NOT INSTALLED | 4.46.3 | added |
| torch | 2.5.0a0+872d972e41.nv24.08 | 2.5.0a0+872d972e41.nv24.08 | no |
| torchvision | 0.20.0a0+afc54f7 | 0.20.0a0+afc54f7 | no |
| numpy | 1.23.5 | 1.23.5 | no |
| tensorrt | 10.3.0 | 10.3.0 | no |
| PIL | 9.0.1 | 9.0.1 | no |
| torch.cuda.is_available() | True | True | no |

### OWL-ViT / Transformers import check

- `from transformers import OwlViTProcessor, OwlViTForObjectDetection, OwlViTImageProcessor` → OK (the OWL-ViT classes NanoOWL depends on import cleanly under transformers 4.46.3). No model weights fetched.

### Stage 0 regression: ONNX -> TensorRT smoke test

- Re-ran `scripts/smoke_onnx_trt.sh`. Result: **PASS** (exit 0).
- Build log contained `&&&& PASSED TensorRT.trtexec [TensorRT v100300]`.
- Run log contained `&&&& PASSED TensorRT.trtexec`; times.json / profile.json / layers.json emitted; `identity.engine` non-empty.
- Conclusion: installing transformers 4.46.3 did not disturb the already-validated ONNX -> TensorRT deployment chain.

### Integrity

- Only `transformers==4.46.3` (plus resolver companions) added to `.venv`. No other package install/upgrade/downgrade, no weights, no engine for NanoOWL, no remote/push. `docs/daily/day01.md` updated + one local commit.
- Stage 1F status: PASS.

## Day 1 / Stage 1G - Isolate Pillow Upgrade + Regression

- Action time: 2026-08-13 10:50 CST
- Upgraded in project `.venv` only: `Pillow==12.3.0` (from 9.0.1). Command: `pip install --upgrade Pillow==12.3.0`.
- Install behavior matched Stage 1E prediction: pip left the system `Pillow 9.0.1` at `/usr/lib/python3/dist-packages` untouched ("Can't uninstall 'Pillow'... outside environment") and installed 12.3.0 into the venv/user overlay; with `include-system-site-packages=true`, import precedence resolves to the newer 12.3.0 copy.
- NOT installed / NOT modified: torch, torchvision, numpy, tensorrt, transformers, torch2trt, timm, accelerate, clip. No OWL-ViT weights downloaded. No torch2trt build. No NanoOWL engine built.

### Post-upgrade verification

| Check | Result |
|---|---|
| `PIL.__version__` | 12.3.0 (asserted == 12.3.0) |
| PIL round-trip (create 16x16 RGB, save PNG+JPEG, reopen, load, size check) | PASS |
| torch | 2.5.0a0+872d972e41.nv24.08 (unchanged from Stage 1F) |
| torchvision | 0.20.0a0+afc54f7 (unchanged) |
| numpy | 1.23.5 (unchanged) |
| tensorrt | 10.3.0 (unchanged) |
| transformers | 4.46.3 (unchanged) |
| torch.cuda.is_available() | True |
| OWL-ViT import (`OwlViTProcessor`, `OwlViTForObjectDetection`) | PASS |

### Stage 0 regression: ONNX -> TensorRT smoke test

- Re-ran `scripts/smoke_onnx_trt.sh`. Result: **PASS** (exit 0).
- Both build and run logs contain `&&&& PASSED TensorRT.trtexec [TensorRT v100300]`; engine non-empty; times/profile/layers emitted.
- Conclusion: the isolated Pillow 9.0.1 -> 12.3.0 upgrade (the upstream UnidentifiedImageError fix) did not disturb the validated ONNX -> TensorRT chain or the transformers 4.46.3 OWL-ViT import path.

### Integrity

- Only `Pillow` changed in `.venv` (9.0.1 -> 12.3.0). No other package install/upgrade/downgrade, no weights, no NanoOWL engine, no remote/push. `docs/daily/day01.md` updated + regenerated `results/smoke/` regression artifacts committed under one local commit.
- Stage 1G status: PASS.

## Day 1 / Stage 1H - Pin torch2trt Upstream (NETWORK_BLOCKED)

- Attempt time: 2026-08-13 10:57 CST
- Input upstream: `https://github.com/NVIDIA-AI-IOT/torch2trt`
- Intended output: Git submodule at `third_party/torch2trt` with the official upstream commit pinned by the parent repository; record upstream URL / branch / HEAD commit / latest commit date / LICENSE, and read README/setup.py for install method + plugin optionality + version pin (record only, no build/install).
- Result: **FAIL - network**. `git submodule add https://github.com/NVIDIA-AI-IOT/torch2trt third_party/torch2trt` failed: `Failed to connect to github.com port 443 after ~131s: Connection timed out`. A standalone `git clone --depth 1` (Stage 1E inspection pass) and the retry here both timed out; `curl --max-time 40 https://github.com/NVIDIA-AI-IOT/torch2trt` returned http_code=000 (timeout) on 3 consecutive attempts.
- Per Stage 1H rule, no mirror or alternate source was used. The upstream URL was not changed.
- Upstream commit hash / branch / latest commit date / LICENSE / README/setup.py install method / plugin optionality / version pin: **unavailable** — the official repository could not be cloned/read from this host at this time.
- Parent repository commit: created only this failure record (commensurate with the Stage 1C network-failure handling); no submodule pin commit was possible because the clone did not succeed.
- Environment integrity: no torch2trt installed, no plugins compiled, no `.venv` change, no torch/torchvision/numpy/TensorRT/CUDA modification, no OWL-ViT weights, no NanoOWL package install, no TensorRT engine built, no remote/push. Parent working tree is clean; `.gitmodules` still contains only the `third_party/nanoowl` entry (no torch2trt partial registration); no `third_party/torch2trt` directory exists.
- Stage 1H status: NETWORK_BLOCKED - GitHub clone timed out (network). Resolved offline via Stage 1H-Fallback below; the submodule-add command was NOT attempted again (no network used).

## Day 1 / Stage 1H-Fallback - Pin Offline torch2trt Upstream

- Fallback time: 2026-08-13, triggered by Stage 1H NETWORK_BLOCKED (GitHub port 443 timed out; no mirror/alternate used in 1H).
- Offline source: `/home/nvidia/torch2trt.tar.gz` — the official torch2trt Git repository pre-staged offline as a full Git archive (includes the complete `.git/` directory).
- Placement: extracted into `third_party/torch2trt`. The `.git` metadata directory was **preserved** (not deleted) so the upstream identity can be verified locally without any network access.
- Pre-extraction state: the prior `third_party/torch2trt` was only an empty gitlink shell — a 74-byte `.git` file pointing at a nonexistent `…/modules/third_party/torch2trt`; no working-tree files, and torch2trt was never registered in `.gitmodules` (which still lists only `third_party/nanoowl`). The shell carried no content and was removed cleanly before extraction.
- Verification (local Git only, **no network used**):
  - origin remote: `https://github.com/NVIDIA-AI-IOT/torch2trt.git` (matches the intended official upstream).
  - HEAD commit hash: `4e820ae31b4e35d59685935223b05b2e11d47b03`
  - branch: `master`
  - latest commit date: `2024-05-03 14:28:37 -0700` (subject: "Merge pull request #922 from jaybdub/dev_trt_10_refactor")
  - working tree clean; 140 tracked files.
- Upstream dependency pinning: the nested repository is kept as a **nested Git repo** (NOT converted to a standard submodule). Standard submodule registration/re-fetch would require network, and per the stage rules no network operation or repository-breaking action is permitted. The upstream is therefore pinned for this parent project by the **recorded commit hash** `4e820ae31b4e35d59685935223b05b2e11d47b03`. `.gitmodules` is left unchanged (still only the `third_party/nanoowl` entry).
- README/setup.py inspection (record only — no install/build):
  - Python library install method: `python setup.py install` (README, Setup Step 1).
  - Plugins: **optional**. README labels plugins "Step 2 (optional)" — built independently via CMake (`cmake -B build . && cmake --build build --target install && ldconfig`). The `--plugins` flag in `setup.py` is deprecated-only and still requires a CUDA/nvinfer compile; it was NOT used.
  - Version pin: `setup.py` declares `version='0.5.0'`; there is no git tag pin and no requirements.txt pin, so the effective version lock for this parent project is the recorded commit hash above.
- Environment integrity: no `pip install`, no torch2trt install, no plugin compilation, no `.venv` modification, no CUDA/TensorRT/PyTorch modification, no model download, no TensorRT engine build, no `git push`, no GitHub upload. Only local extraction, local Git verification, this documentation update, and one local parent commit were performed.
- Stage 1H-Fallback status: PASS.
