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
        
        import ollama

        model_code = self._get_ollama_model_code()

        try:
            response = self.model.chat(
                model=model_code,
                messages=[{
                    'role': 'user',
                    'content': question,
                    'images': [image_path]
                }],
                options={
                    'temperature': 0
                }
            )
        except ollama._types.ResponseError as e:
            if 'this model is missing data required for image input' in str(e):
                self.log_fatal(f'This ollama model ("{model_code}") does not support image inputs. Please check model doc on ollama.com')
            raise(e)

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
        from ollama import Client, _types
        self.client = Client(host=BVQA_OLLAMA_HOST)
        self.model = self.client

        self.stop_if_ollama_not_running()

        self.pull_ollama_model()


        # load the model into memory so get_compute_info get stats.
        # and we separate the load from the actual inference in the logs.
        try:
            res = self.model.generate(model=self._get_ollama_model_code(), prompt='Just say "yes". Nothing else.')
        except _types.ResponseError as e:
            print('H0'*40)
            raise e

        return self.model

    def stop_if_ollama_not_running(self):
        self.get_ollama_models()

    def is_ollama_model_pulled(self):
        ret = False
        model_code = self._get_ollama_model_code()
        models = self.get_ollama_models()
        matching_models = [
            model
            for model
            in models
            if model.model == model_code
        ]
        return bool(matching_models)

    def pull_ollama_model(self):
        if not self.is_ollama_model_pulled():
            import ollama
            model_code = self._get_ollama_model_code()
            self.log(f'Pulling model "{model_code}" from Ollama repository...')
            try:
                res = self.client.pull(model_code)
            except ollama._types.ResponseError as e:
                self.log_fatal(f'failed to pull model "{model_code}" from Ollama repository. {e}')

    def get_ollama_models(self):
        ret = []
        try:
            res = self.client.list()
            ret = res.models
        except ConnectionError as e:
            self.log_fatal(f'{e} ; BVQA_OLLAMA_HOST="{BVQA_OLLAMA_HOST}"')
        return ret

    def get_compute_info(self):
        ret = {
            'type': 'ollama',
            'desc': 'ollama',
            'size': 0
        }
        model = self.get_model()

        '''
        # models=[Model(model='gemma3:4b', name='gemma3:4b', digest='a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a', expires_at=datetime.datetime(2025, 7, 18, 14, 3, 39, 405547, tzinfo=TzInfo(+01:00)), size=6169209844, size_vram=3700725812, details=ModelDetails(parent_model='', format='gguf', family='gemma3', families=['gemma3'], parameter_size='4.3B', quantization_level='Q4_K_M'))]
        res = self.client.ps()
        print(res)
        exit(1)
        '''

        if model:
            res = model.ps()
            for m in res.models:
                if self.model_id in m.name:
                    ret['size'] = m.size
                    break

        return ret
