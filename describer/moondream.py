# Keep all slow imports (e.g. torch) inside methods
from .base import ImageDescriber
from PIL import Image
from pathlib import Path
import datetime, time

'''
On 22nd Feb 2025, 
HF transformers API can only run the model on GPU.
Because there is a bug with the image encoding on CPU.

Moondream API can run the model on CPU.
Because it GPU support is forthcoming.
'''
MODEL_ID = 'moondream-0_5b-int8'
MODEL_VERSION = '9dddae84d54db4ac56fe37817aeaeb502ed083e2'
DEFAULT_VERSION = {
    # Default on CPU, uses moondream API
    MODEL_ID: MODEL_VERSION,
    'moondream-2b-int8': MODEL_VERSION,
    # Default on GPU, uses HF transformers API
    'vikhyatk/moondream2': '2025-01-09',
}

# only used with the CPU version of Moondream API, without transformers
MODELS_PATH = Path('models')

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
        if not model_version:
            model_version = DEFAULT_VERSION.get(model_id, '')
            if not model_version:
                raise Exception(f'Please specify the model version for {model_id}.')

        super().__init__(model_id, model_version)
        self.model = None
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
        self.model = None

        # import torch
        # is_cuda_available = torch.cuda.is_available()

        if self._does_model_need_transformers():
            import torch
            try:
                self._new_model(use_cuda=True, use_attention=self.optimise)
            except torch.cuda.OutOfMemoryError as e:
                print('ERROR: Model exceeds VRAM')
            except RuntimeError as e:
                print(f'ERROR: Unknown error while using CUDA: "{e}"')
        else:
            self._new_model()

        return self.model

    def _does_model_need_transformers(self):
        return '/' in self.model_id
    
    def _new_model(self, use_cuda=False, use_attention=False):
        if use_cuda:
            from transformers import AutoModelForCausalLM
            import torch

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.model_version,    
                trust_remote_code=True,
                device_map="cuda" if use_attention or use_cuda else None,
                # torch_dtype = torch.float16 if use_cuda else None,
                attn_implementation = "flash_attention_2" if use_attention else None
            )
        else:
            import moondream as md
            model_path = self._download_model()
            if model_path:
                self.model = md.vl(model=str(model_path))

    def _download_model(self):
        # 1. create 'models' subdirectory if it doesn't exist
        # 2. download compressed model from MODEL_URL into 'models' subdirectory (MODEL_PATH)
        # 3. decompress the downloaded gz file
        # 4. test that we have a model file at MODEL_PATH
        # 5. if so delete the compressed model file
        import os
        import requests
        import gzip

        ret = None

        model_filename = f'{self.model_id}.mf'
        model_url = f'https://huggingface.co/vikhyatk/moondream2/resolve/{self.model_version}/{model_filename}.gz?download=true'
        model_path = MODELS_PATH / f'{self.model_id}-{self.model_version}.mf'

        if model_path.exists(): return model_path
        MODELS_PATH.mkdir(parents=True, exist_ok=True)

        print(f'INFO: downloading moondream model {self.model_id}:{self.model_version} ...')

        model_path_gz = model_path.with_suffix('.gz')
        if not model_path_gz.exists():
            response = requests.get(model_url)
            with open(model_path_gz, 'wb') as f:
                f.write(response.content)

        with gzip.open(model_path_gz, 'rb') as f_in:
            with open(model_path, 'wb') as f_out:
                f_out.writelines(f_in)

        if not model_path.exists():
            raise FileNotFoundError(f'Model file not found at {model_path}')
        else:
            os.remove(model_path_gz)
            ret = model_path
        
        return ret

