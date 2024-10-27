# Batch visual question answering (BVQA)

This python command line tool lets you ask a series of questions 
to a visual language model about a collection of images.
It saves the answers in json files (one file per image).

## Features

* suppport for parallel processing
  * simply start additional instances on same image collection
  * can resume processing after abrupt interruption
  * tested on SLURM environment (KCL ER HPC)
* designed to work with different vision language models
  * which also allows to compare/benchmark different models
* granual caching: only ask question again if prompt or model has changed
* [coming soon] export answers to HTML for manual review

## Requirements

Follow the [build instructions](build/README.md).

The first time the script runs it will download the visual language model that answers the questions.
Currently, this is [moondream2](https://github.com/vikhyat/moondream) because it's light, fast & performs generally well.

Although a GPU is not mandatory for the moondream model, processing will be very slow without it.

## Usage

By default the root folder for all the input and output is /data.

### Prepare your input

* **/data/images**: copy your input images (*.jpg) anywhere under that folder
* **/data/questions.json**: your questions (see example in [/test/data/questions.json](/test/data/questions.json))

### Generate descriptions

`python bvqa.py describe`

### Output

* **/data/answers/**: contains the generated answers. Each json file contains all the answers for an image
* **/data/describe.log**: a log of the processing for monitoring and performance purpose

## Options

### Command line

For more additional options see:

`python bvqa.py -h`

Such as:

* -m to select a different vision language model (VLM) / describer
* -f to filter which images are processed
* -q to filter which questions to ask
* -R to use a different root folder for your data

### Config file

[TODO]

## Models

### Supported

#### [moondream](https://huggingface.co/vikhyatk/moondream2) (1.87b)

default model, decent responses for general questions, quite fast, even on CPU. Speed: ~9s/question/image on i9 CPU. ~1.7s/q/i on 1080ti (~5x faster; 72s for 2 img x 21 qst). Deterministic. 

#### [qwen-vl](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4) (1.07B)

* **GPU only model with higher memory requirements**
* **Harder to install dependencies**
* generally quite good at "focused" OCR/HTR 
* able to work on any image resolution (no downsizing)
* Tried the int4 ~5 x slower than MD by default. With higher mem consumption.

### Non supported models

* llava-next [TODO]: 0.5b needs access to gated llama-3-8b model on HF
* Phi-3.5-vision-instruct (4.15B) [TRIED]: demo code out of memory on A30 (24GB) although model is 4.15B x BF16. Requires 50GB VRAM!
* Vila [MAYBE]: docker not building, can't find instructions on how to run model with python
* HuggingFaceM4/Idefics3-8B-Llama3 [MAYBE]: Requires ~24GB VRAM.

