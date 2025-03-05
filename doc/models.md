## Models

### Supported

* Moondream2
* SmolVLM
* Qwen2-VL
* LLama3.2-vision (via Ollama)
* minicpm-v (via Ollama)

#### [moondream](https://huggingface.co/vikhyatk/moondream2) (1.87b, FP16)

Default model, decent responses for general questions, quite fast, even on CPU. Deterministic.

* Downsampling: max 756 pixels any side
* Minimum GPU VRAM: 4.5GB
* CPU (i7 12th gen, 16 cpus): 5 to 40s / question (short to long)
* A30: 1 to 9s / qst

Pros:
* very small, fast and robust
* good for general purpose questions
* good at counting
* trained for OCR

Cons:
* No chat mode? (i.e. pass one image & question, then add more questions)
* Very keen to provide general/contextual descriptions despite instructions to focus on particular element in the image

#### [Smol](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)

Alternative to Moondream as it's only slightly bigger.

* Downsampling: max 1536 pixels any side (N=5 for N*384, N can be adjusted)
* Minimum GPU VRAM: 6.2GB
* CPU (i7 12th gen, 16 cpus): 
* A30: 

Pros:
* can be very good at detailed description

Cons:
* prone to repetition with default settings
* can make up details when asked for long description and there's nothing new to say

TODO:
* test on film frames
* test on OCR/forms
* optimise for CPU
* support for 4bits
* test chat mode

#### [qwen-vl](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4) (1.07B)

* **GPU only model with higher memory requirements**
* **Harder to install dependencies**
* generally quite good at "focused" OCR/HTR 
* able to work on any image resolution (no downsizing)
* Tried the int4, ~5 x slower than moondream by default. With higher mem consumption

We got noticably better results on form understanding with the 7b+ model than the 2b.
However the VRAM requirement can be very high, especially on larger images.

2b model:

* VRAM: 16GB
* A30: 7 to 40s / qst

7b model:

* VRAM: >40GB
* A30: ?

#### [ollama](https://ollama.com/)

This describer sends questions to an Ollama instance.
Ollama loads and runs (vision) language models in a separate on the same or remote machine.
bvqa sends prompts to Ollama via its web API.

The default model in the ollama describer is [llama3.2-vision:11b](https://ollama.com/library/llama3.2-vision:11b). 
You can use a different model and version with the `-m` and `-v` arguments.
List of [VLMs supported by Ollama](https://ollama.com/search?c=vision&o=newest).

Consult the ollama website to learn how to install it 
and download models.

Note that bvqa communicate with Ollama on port 11435.
Ollama, by default, listens on port 11434.
Use `ssh -L 11435:H:11434` to connect the two 
(where H is the host Ollama is running from).

Advantages of Ollama:

* Ollama is straightforward to install
* it can run remotely by offloading the heavy model computation to a remote GPU
* this lets you run bvqa locally within a comfortable/familiar development environemnt,
* simplifies the local stack, keeping it light
* Ollama comes with many quantised models for efficient execution on more modest compute
* speeds up bvqa startup time by keeping the model in memory between runs
* good suppoort for multiple types of computes (including hybrid, e.g. CPU + GPU)

Disadvatanges:

* it only supports a few VLMs and lags behind huggingface
* e.g. Moondream on Ollama is 6 month behind latest version
* complicates prallel/distributed processing over multiple machines (because we need to run multiple ollama services)

[Ollama can be installed & instantiated manually without sudo](https://github.com/ollama/ollama/blob/main/docs/linux.md#manual-install).

TODO: for parallelism a sbatch should provided to launch ollama on unique port and pass it to bvqa.

#### [llama3.2-vision](https://ollama.com/library/llama3.2-vision)

Supported via the ollama describer.
Good at following more precise instructions.

11b (4bits):

* Downsampling: max  1120x1120 pixels
* Minimum GPU VRAM: 12GB
* Context: 128k
* Output tokens: 2048

#### [minicpm-v](https://ollama.com/library/minicpm-v)

Supported via the ollama describer.

MiniCPM-V 2.6, 8b:

* Downsampling: any aspect ratio and up to 1.8 million pixels (e.g., 1344x1344).
* Minimum GPU VRAM: 7GB
* Context: ?
* Output tokens: ?

### Non supported models

Regularly check [Huggingface VLM leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) and [OpenCompass Leaderboard](https://mmbench.opencompass.org.cn/leaderboard) for new candidates.

* [molmo](https://molmoai.com/#features): open source trained on well curated dataset with good general understanding, comes in 72b, 7b (-o is most open, -d perform better) and 1b.
* [InternVL](https://huggingface.co/collections/OpenGVLab/internvl25-673e1019b66e2218f68d7c1c): v. good performance in benchmarks
* llava-next [NO]: 0.5b needs access to gated llama-3-8b model on HF
* Vila [MAYBE]: docker not building, can't find instructions on how to run model with python
* HuggingFaceM4/Idefics3-8B-Llama3 [MAYBE]: Requires ~24GB VRAM
* Phi-3.5-vision-instruct (4.15B) [TRIED]: demo code out of memory on A30 (24GB) although model is 4.15B x BF16. Requires 50GB VRAM!
* PaliGemma ?
* florence [NO]: it doesn't support free prompt tasks, only set tasks such as global/region OCR or CAPTION

### Adding support for a new type of VLM to this tool

In short, you would need to create a new describer module and class under /describer package. 
It should extend from ImageDescriber (in describe/base.py) and implement answer_question() and _init_model().
