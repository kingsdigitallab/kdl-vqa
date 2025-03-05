# Keep all slow imports (e.g. torch) inside methods
from .base import ImageDescriber
from PIL import Image
from pathlib import Path
import datetime, time
import os

# https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
MODEL_ID = 'llama3.2-vision'
MODEL_VERSION = '11b'
BVQA_OLLAMA_HOST = os.getenv('BVQA_OLLAMA_HOST', 'http://localhost:11434')

class Ollama(ImageDescriber):
    """Image description using a model served by Ollama.

    List of avaible models at:
    
    https://ollama.com/search?c=vision

    TODO: optimise. At the moment we are reading 
    and sending the image for each question.

    Some models, like llama work in chat mode.
    In that mode image only need to be sent in the first question.
    Others, like moondream, don't.
    But there's no way to know with the API.
    """    
    
    def __init__(self, model_id='', model_version=''):
        model_id = model_id or MODEL_ID
        if model_id == MODEL_ID and not model_version:
            model_version = MODEL_VERSION
        super().__init__(model_id, model_version)
        self.cache = {
            'image_path': None,
            'image_encoding': None,
        }

    def _get_ollama_model_code(self):
        ret = self.model_id
        if self.model_version:
            ret = f'{self.model_id}:{self.model_version}'
        return ret

    def answer_question(self, image_path, question):
        if not self.model:
            self._init_model()

        response = self.model.chat(
            model=self._get_ollama_model_code(),
            messages=[{
                'role': 'user',
                'content': question,
                'images': [image_path]
            }],
            options={
                'temperature': 0
            }
        )

        # example of a response
        '''
        {
            'model': 'llama3.2-vision', 
            'created_at': '2024-11-18T16:09:55.388731028Z', 
            'message': {
                'role': 'assistant', 
                'content': 'Yes.'
            }, 
            'done_reason': 'stop', 
            'done': True, 
            'total_duration': 5967723103, 
            'load_duration': 3842134351, 
            'prompt_eval_count': 31, 
            'prompt_eval_duration': 1389000000, 
            'eval_count': 3, 
            'eval_duration': 75000000
        }
        '''

        return response['message']['content']

    def _init_model(self):
        from ollama import Client
        self.model = Client(host=BVQA_OLLAMA_HOST)
        # load the model into memory so get_compute_info get stats.
        # and we separate the load from the actual inference in the logs.
        res = self.model.generate(model=self._get_ollama_model_code(), prompt='Just say "yes". Nothing else.')
        return self.model

    def get_compute_info(self):
        ret = {
            'type': 'ollama',
            'desc': 'ollama',
            'size': 0
        }
        model = self.get_model()

        if model:
            res = model.ps()
            for m in res.models:
                if self.model_id in m.name:
                    ret['size'] = m.size
                    break

        return ret
