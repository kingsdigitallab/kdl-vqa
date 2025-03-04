# get the parent of the folder where this script sits
BVQA_ROOT="$(dirname "$0")/../.."
docker run --gpus=all -it --rm -v "$BVQA_ROOT:/app" -w /app python:3.12-slim bash test/requirements/all.bash
