# Keep all slow imports (e.g. torch) inside methods
from .base import ImageDescriber
from PIL import Image
from pathlib import Path
import datetime, time

MODEL_ID = 'HuggingFaceTB/SmolVLM-Instruct'
MODEL_VERSION = ''
MAX_NEW_TOKENS = 512


class SmolVLM(ImageDescriber):
    """Image description using SmolVLM model.

    https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct
    2.25B params, BF16
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
            from transformers.image_utils import load_image
            self.cache['image_path'] = image_path
            self.cache['image_encoding'] = load_image(image_path)

        # Create input messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
        ]

        # Prepare inputs
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[self.cache['image_encoding']], return_tensors="pt")
        
        # todo: base fct for that device_type
        device_type = next(self.model.parameters()).device.type
        inputs = inputs.to(device_type)

        # Generate outputs
        generated_ids = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return generated_texts[0]

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
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        return self.model
    
    def _new_model(self, use_cuda=False, use_attention=False):
        from transformers import AutoModelForVision2Seq
        import torch

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2" if use_attention else "eager",
        )
        if use_cuda:
            self.model = self.model.to("cuda")
        
    def _encode_image(self, image_path):
        pass
