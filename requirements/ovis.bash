# GPU processing with transformers library
# https://huggingface.co/AIDC-AI/Ovis2-1B#usage
. torch.bash
ltt install -r ovis.txt

# Commented out at the build 1) takes an eternity and 2) has harder requirements
# pip install flash-attn==2.7.0.post2 --no-build-isolation
FLASH_ATTN_VERSION="2.7.4.post1"
# this method requires nvcc
# CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+')
CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
TORCH_VERSION=$(python -c "import torch; print(f'{torch.__version__.split('.')[0]}.{torch.__version__.split('.')[1]}')");
FLASH_ATTN_WHEEL="flash_attn-${FLASH_ATTN_VERSION}+cu${CUDA_VERSION}torch${TORCH_VERSION}cxx11abiTRUE-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-linux_x86_64.whl"
wget "https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/$FLASH_ATTN_WHEEL" -P /tmp/ -N
pip install "/tmp/$FLASH_ATTN_WHEEL"
