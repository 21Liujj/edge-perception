# Source this file before running CUDA command-line tools.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/targets/aarch64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
