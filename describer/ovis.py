# Keep all slow imports (e.g. torch) inside methods
from .base import ImageDescriber
# from PIL import Image
from pathlib import Path
import datetime, time

MODEL_ID = 'AIDC-AI/Ovis2-1B'
MODEL_VERSION = ''
MAX_NEW_TOKENS = 512


class Ovis(ImageDescriber):
    """Image description using SmolVLM model.

    https://huggingface.co/AIDC-AI/Ovis2-1B
    1.27B params, BF16
    """    
    
    def __init__(self, model_id='', model_version=''):
        model_id = model_id or MODEL_ID
        if model_id == MODEL_ID and not model_version:
            model_version = MODEL_VERSION
        super().__init__(model_id, model_version)
        self.model = None
        self.text_tokenizer = None
        self.visual_tokenizer = None
        self.cache = {
            'image_path': None,
            'image_encoding': None,
        }

    def answer_question(self, image_path, question):
        if not self.model:
            self._init_model()

        if image_path != self.cache['image_path']:
            from PIL import Image
            self.cache['image_path'] = image_path
            self.cache['image_encoding'] = Image.open(image_path)

        import torch

        query = f'<image>\n{question}'
        images = [self.cache['image_encoding']]
        max_partition = 9

        prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, images, max_partition=max_partition)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)
        pixel_values = [pixel_values]

        # generate output
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True
            )
            output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
            ret = output

        return ret

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

        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        return self.model
    
    def _new_model(self, use_cuda=False, use_attention=False):
        from transformers import AutoModelForCausalLM
        import torch

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            multimodal_max_length=32768,
            trust_remote_code=True
        )
        if use_cuda:
            self.model = self.model.cuda()

        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        
    def _encode_image(self, image_path):
        pass
