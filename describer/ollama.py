# Keep all slow imports (e.g. torch) inside methods
from .base import ImageDescriber
from PIL import Image
from pathlib import Path
import datetime, time

# https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
MODEL_ID = 'llama3.2-vision'
MODEL_VERSION = '11b'
OLLAMA_HOST = 'http://localhost:11435'

class Ollama(ImageDescriber):
    """Image description using a model served by Ollama.

    https://github.com/ollama/ollama-python

    Note Ollama listens to port 11434 by default.
    This describer sends a requests to OLLAMA_HOST.
    Which is on a different port.

    For that to work you need to run ssh -L11435:H:11434

    Where H is the host that runs Ollama.

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

    def answer_question(self, image_path, question):
        if not self.model:
            self._init_model()

        model_arg = self.model_id
        if self.model_version:
            model_arg = f'{self.model_id}:{self.model_version}'

        response = self.model.chat(
            model=model_arg,
            messages=[{
                'role': 'user',
                'content': question,
                'images': [image_path]
            }]
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
        self.model = Client(host=OLLAMA_HOST)
        return self.model

    def get_compute_info(self) -> str:
        return 'Unknown remote compute for Ollama'
    