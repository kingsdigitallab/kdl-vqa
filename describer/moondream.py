# Keep all slow imports (e.g. torch) inside methods
from .base import ImageDescriber
from PIL import Image
from pathlib import Path
import datetime, time

MODEL_ID = 'vikhyatk/moondream2'
MODEL_VERSION = '2025-01-09'


class Moondream(ImageDescriber):
    """Image description using Moondeam2 model.

    https://huggingface.co/vikhyatk/moondream2
    1.87B params, FP16
    https://github.com/vikhyat/moondream

    Features:
    * A small & fast model that is easy to install, deploy & use.
    * Runs on CPU or GPU (~4.5GB VRAM).
    * max_new_tokens is 256 by default
    * accepted image sizes: 378/756 x 378/756
    * Text model supports flash attention 2.
    * Good at OCR as well.
    * Ability to return bounding boxes ('Bounding box: license plate').
    * Good at counting.
    * Well maintained, with regular upgrades.
    * License: Apache 2.0
    * Deterministic
    * TBC: upcoming faster cpu version with ggml/
        https://github.com/vikhyat/moondream/tree/moondream-ggml/moondream-ggml

    Optimisations (see set_optimisation() and -o flag in describe.py):
    * use flash attention ([but higher requirements](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#nvidia-cuda-support))

    Limitations:
    * No quantised models available.
    * Each prompt must supply a single question about a single image.
    * No memory/chat mode.
    * So there's not much way to optimise for lots of qst on same image.
    * No video. Single image only.
    * moondream.batch_answer() isn't faster than calling answer_question multiple times
    """    
    
    def __init__(self, model_id='', model_version=''):
        model_id = model_id or MODEL_ID
        if model_id == MODEL_ID and not model_version:
            model_version = MODEL_VERSION
        super().__init__(model_id, model_version)
        self.model = None
        self.tokenizer = None
        self.cache = {
            'image_path': None,
            'image_encoding': None,
        }

    def answer_question(self, image_path, question):
        if not self.model:
            self._init_model()

        if image_path != self.cache['image_path']:
            self.cache['image_path'] = image_path
            image = Image.open(image_path)
            self.cache['image_encoding'] = self.model.encode_image(image)

        return self.model.answer_question(self.cache['image_encoding'], question, self.tokenizer)

    def _init_model(self):
        import torch
        self.model = None

        is_cuda_available = torch.cuda.is_available()
        if is_cuda_available:
            try:
                self._new_model(use_cuda=True, use_attention=self.optimise)
            except torch.cuda.OutOfMemoryError as e:
                print('WARNING: Model exceeds VRAM')
            except RuntimeError as e:
                print(f'WARNING: Unknown error while using CUDA: "{e}"')

        if self.model is None:
            print('WARNING: running model on CPU')
            self._new_model()

        # why no .to(X) ?
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision = self.model_version)

        return self.model
    
    def _new_model(self, use_cuda=False, use_attention=False):
        from transformers import AutoModelForCausalLM
        import torch

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            revision=self.model_version,    
            device_map="cuda" if use_attention else None,
            torch_dtype = torch.float16 if use_cuda else None,
            attn_implementation = "flash_attention_2" if use_attention else None
        )
        if use_cuda and not use_attention:
            self.model = self.model.to("cuda")
        
    def _encode_image(self, image_path):
        pass
