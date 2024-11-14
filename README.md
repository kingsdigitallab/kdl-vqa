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
Currently, this is [moondream2](https://github.com/vikhyat/moondream) because it's light, fast & performs generally well and is well maintained.

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

* -d to select a different describer (moondream or qwen-vl at the moment)
* -m to specify which exact model the describer should use (e.g. vikhyatk/moondream2)
* -v to specify the version/revision of the model (e.g. 2024-08-26)
* -f to filter which image is processed
* -q to filter which questions to ask
* -R to use a different root folder for your data

### Config file

[TODO]

## Caching

The tool will not ask a question again if an answer has been saved.
It will ask it again only if the question or model has changed. 
This allows you to iteratively reformulate one question at a time 
and only that question will be processed on your image collection. 
Which is convenient considering how much models can be sensitive to the phrasing of a prompt. 
You can combine this with the -f option to test on a few images only.

The -r option tells the tool to ignore the cache. 
When supplied, it will always ask the questions again. 
This is useful in the case where you want to compare the performance between different computing devices (e.g. Nvidia A100 vs L40s GPUs) to estimate the total duration on your entire collection.

## Parallelism

To speed up processing you can run multiple instances of the tool in parallel. 
For this to work they need to write in the same `answers` folder. 
Each instance locks the image by writing a timestamp in the answer file. 
Other instances will skip the image when the timestamp is no older than two minutes.

### SLURM HPC

Simplest approach is to distribute the instances among different GPUs.

Following command on SLURM environment sends two instances (-n 2) to a compute node each instance will use 4 cpus, 8GB of RAM and one A30 GPU:

`srun -p interruptible_gpu -c 4 --mem-per-gpu 8G --gpus-per-task 1 --constraint a30 -n 2 python bvqa.py describe`

You can keep adding more instances with further calls to `srun` ([srun doc](https://slurm.schedmd.com/srun.html)).

[TODO: provide sbatch script]

### Dedicated machine

On a dedicated machine with multiple GPUs, you can launch each instance on a specific GPU like this:

`CUDA_VISIBLE_DEVICES=0 nohup python bvqa.py describe &`

`CUDA_VISIBLE_DEVICES=1 nohup python bvqa.py describe &`

If a single instance uses less than 50% of the GPU VRAM and processing (use `nvidia-smi dmon` to check) 
and less than 50% of CPU & RAM then you can send another instance on the same GPU.

## Tips

Finding the right model and prompts to get good answers is a matter of trial and errors. 
It requires iteratively crafting the questions to get the desired form and accuracy of responses. 
Some models need some nudging. 
And some questions will never be satisfactorily answered by a given model or any curent model. 
Knowing how to rephrase, reframe or simply stop is a bit of an art.

A recommended method is to work one question at a time with a handful of diverse images. 
Engineer the prompt to optimise accuracy. If it is high enough, iterate over a slightly larger sample of images. 
If you are confident the level of error is tolerable and your sample representative enough of the whole collection 
then the question is worth submitting to all the images.

It is more computationally efficient to prepare all your questions before sending them to the entire collection. 
Running one question at a time over N images Q times is much slower than running the script once with Q questions over N images.

After running your questions on a larger proportion of your collection, you might want to spot check the responses here and there to get a sense of how good/usable they are.

As prompt engineering is usually very model-specific, moving to another model can be very disruptive. 
It aways mean reassessing the answers and often means reformulating many questions from scratch.

## Models

### Supported

#### [moondream](https://huggingface.co/vikhyatk/moondream2) (1.87b, FP16)

default model, decent responses for general questions, quite fast, even on CPU. Speed: ~9s/question/image on i9 CPU. ~1.7s/q/i on 1080ti (~5x faster; 72s for 2 img x 21 qst). Deterministic.

* Downsampling: max 756 pixels any side
* Minimum GPU VRAM: 8GB
* CPU (i7 12th gen, 16 cpus): 5 to 40s / question (short to long)
* A30: 1 to 9s / qst

#### [qwen-vl](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4) (1.07B)

* **GPU only model with higher memory requirements**
* **Harder to install dependencies**
* generally quite good at "focused" OCR/HTR 
* able to work on any image resolution (no downsizing)
* Tried the int4 ~5 x slower than MD by default. With higher mem consumption.

* VRAM: 16GB
* A30: 7 to 40s / qst

### Non supported models

Regularly check [Huggingface VLM leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) and [OpenCompass Leaderboard](https://mmbench.opencompass.org.cn/leaderboard) for new candidates.

* llama3.2-11B-Vision-Instruct [MAYBE]: 
* llava-next [MAYBE]: 0.5b needs access to gated llama-3-8b model on HF
* Vila [MAYBE]: docker not building, can't find instructions on how to run model with python
* HuggingFaceM4/Idefics3-8B-Llama3 [MAYBE]: Requires ~24GB VRAM
* Phi-3.5-vision-instruct (4.15B) [TRIED]: demo code out of memory on A30 (24GB) although model is 4.15B x BF16. Requires 50GB VRAM!
* PaliGemma ?

### Adding support for a new type of VLM to this tool

[TODO]

In short, you would need to create a new describer module and class under /describer package. 
It should extend from ImageDescriber (in describe/base.py) and implement answer_question() and get_name().

### External references

* [Vision-Language Models for Vision Tasks: A Survey, 2024](https://arxiv.org/abs/2304.00685)
* [Abdallah, A., Eberharter, D., Pfister, Z. et al. A survey of recent approaches to form understanding in scanned documents. Artif Intell Rev 57, 342 (2024). ](https://link.springer.com/article/10.1007/s10462-024-11000-0#Sec12)

