# kdl-vqa
Python tool for batch visual question answering.

Purpose: Ask a series of questions to visual language about a collection of images.

## Features

* suppport for parallel processing
  * simply start additional instances on same image collection
  * can resume processing after abrupt interruption
  * tested on SLURM environment (KCL ER HPC)
* designed to work with different vision language models
  * which also allows to compare/benchmark different models
* granual caching: only ask question again if prompt or model has changed
* can select a different model for each question
* [coming soon] export answers to HTML for manual review

## Requirements & set up

GPU will speed up processing but is optional. 

* [Install `poetry`](https://python-poetry.org/docs/#installing-with-pipx).
* cd into this folder
* `poetry install`: to create the virtual environment & install dependent packages
* `poetry run python describe.py`: to describe a set of images

The first time the script runs it will download the visual language model that answers the questions.
Currently, this is [moondream2](https://github.com/vikhyat/moondream) because it's light, fast & performs generally well.

## Input

* Images (1920x1080): currently the images under the sample folder.
* Questions: in `questions.py` 

## Output

* A JSON file for each frame, with the responses for the model;

(Temporarily disabled: a HTML file with all frames and answers)

## Models

### Moondream2

* Moondream2 (1.87b) [SUPPORTED]: default model, decent responses for general questions, quite fast, even on CPU. Speed: ~9s/question/image on i9 CPU. ~1.7s/q/i on 1080ti (~5x faster; 72s for 2 img x 21 qst). Deterministic. 
* llava-next [TODO]: 0.5b needs access to gated llama-3-8b model on HF
* Phi-3.5-vision-instruct (4.15B) [TRIED]: demo code out of memory on A30 (24GB) although model is 4.15B x BF16. Requires 50GB VRAM!
* Vila [MAYBE]: docker not building, can't find instructions on how to run model with python
* HuggingFaceM4/Idefics3-8B-Llama3 [MAYBE]: Requires ~24GB VRAM.
* Qwen2-VL-2B-Instruct-GPTQ-Int4 (1.07B) [TRIED]: Tried the int4 ~5 x slower than MD by default. With higher mem consumption. Does NOT run on CPU. Harder to install dependencies.

### Qwen-VL

[TODO]

## Optimisations

If you have a machine with one or two GPUs you can run multiple instance of describe.py in parallel, they should be able to answer questions about separate images.

[TODO]
