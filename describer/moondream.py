# Keep all slow imports (e.g. torch) inside methods
from .base import ImageDescriber
from PIL import Image
from pathlib import Path
import datetime, time
import moondream as md

# only used with transformers
MODEL_ID = 'vikhyatk/moondream2'
MODEL_VERSION = '2025-01-09'

# only used with the CPU version of Moondream API, without transformers
MODELS_PATH = Path('models')
MODEL_FILE_NAME = 'moondream-2b-int8.mf'
MODEL_URL = f'https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/{MODEL_FILE_NAME}.gz?download=true'
MODEL_PATH = MODELS_PATH / MODEL_FILE_NAME

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
        self.uses_transformers = False
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

        return self.model.query(self.cache['image_encoding'], question)['answer'].strip()

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
        # from transformers import AutoTokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision = self.model_version)

        return self.model
    
    def _new_model(self, use_cuda=False, use_attention=False):
        if use_cuda:
            self.uses_transformers = True
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
        else:
            self.uses_transformers = False
            self._download_model()
            self.model = md.vl(model=str(MODEL_PATH))

    def _download_model(self):
        # 1. create 'models' subdirectory if it doesn't exist
        # 2. download compressed model from MODEL_URL into 'models' subdirectory (MODEL_PATH)
        # 3. decompress the downloaded gz file
        # 4. test that we have a model file at MODEL_PATH
        # 5. if so delete the compressed model file
        import os
        import requests
        import gzip

        ret = False

        if MODEL_PATH.exists(): return True
        MODELS_PATH.mkdir(parents=True, exist_ok=True)

        print('INFO: downloading moondream model...')

        model_path_gz = MODEL_PATH.with_suffix('.gz')
        if not model_path_gz.exists():
            response = requests.get(MODEL_URL)
            with open(model_path_gz, 'wb') as f:
                f.write(response.content)

        with gzip.open(model_path_gz, 'rb') as f_in:
            with open(MODEL_PATH, 'wb') as f_out:
                f_out.writelines(f_in)

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f'Model file not found at {model_path}')
        else:
            os.remove(model_path_gz)
            ret = True
        
        return ret

    def _encode_image(self, image_path):
        pass

    def get_name(self) -> str:
        ret = super().get_name()
        if not self.uses_transformers:
            ret = str(MODEL_PATH)
        return ret


