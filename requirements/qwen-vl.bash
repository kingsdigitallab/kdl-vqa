# GPU processing with transformers library
. torch.bash
ltt install -r qwen-vl.txt

# Needed for smaller/faster gptq models provided by qwen.
# TODO: replace with gptqmodel once it is easier to install
# known issues (2025-03-04):
# - won't compile with gcc >13
# - won't work with cuda 12.6+?
pip install auto-gptq --no-build-isolation

