# Keep all slow imports (e.g. torch) inside methods
from .base import ImageDescriber
# from PIL import Image
from pathlib import Path
import datetime, time
import torch
import re

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4"
#MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
#allenai/olmOCR-7B-0225-preview
MODEL_VERSION = ''
MAX_NEW_TOKENS = 512
# TODO: external parameter?
MAX_PIXELS = 3000 * 3000
# MAX_PIXELS = 30 * 30

class QwenVL(ImageDescriber):
    """Image description using Qwen2-VL model.

    https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4

    1.07B params, int4
    """    
    
    def __init__(self, model_id='', model_version=''):
        super().__init__(model_id or MODEL_ID, model_version or MODEL_VERSION)
        self.model = None
        self.processor = None
        self.cache = {
            'image_path': None,
            'image_encoding': None,
        }

    # disabled: faster but quality will drop as number of qst increase
    def answer_questions_unreliable(self, image_path: str, questions: dict) -> dict:
        """Returns answers to all questions 
        about an image in image_path.
        """
        ret = {}
        question_keys = ', '.join(questions.keys())
        self.log(f'question - before - {question_keys}')

        keys = list(questions.keys())
        question = 'Answer all the following questions.\n'
        i = 0
        for k in keys:
            i += 1
            question += f'{i}. {questions[k]}\n'
        res = self.answer_question(image_path, question)

        # print(res)

        answers = re.findall(r'(?m)^(\d+).\s*(.*)\s*$', res)
        for i, key in enumerate(keys):
            if i + 1 > len(answers):
                answer = 'NOANSWER'
            else:
                answer = answers[i][1]
            ret[key] = answer

        self.log(f'question - after - {question_keys}')

        # print(ret)
        
        return ret


    def answer_question(self, image_path, question):
        from qwen_vl_utils import process_vision_info

        if not self.model:
            self._init_model()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "min_pixel": 0,
                        "max_pixel": MAX_PIXELS,
                        # "resized_height": 425,
                        # "resized_width": 756,                    
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self.cache['image_path'] != image_path:
            image_inputs, video_inputs = process_vision_info(messages)
            self.cache['image_encoding'] = [image_inputs, video_inputs]
        else:
            image_inputs, video_inputs = self.cache['image_encoding']
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        comp_info = self.get_compute_info()
        if comp_info['type'] == 'cuda':
            inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        ret = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return ret[0]

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
            raise Exception('GPU is needed for this model.')

        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        return self.model

    def _new_model(self, use_cuda=False, use_attention=False):
        if '2.5' in MODEL_ID:
            from transformers import Qwen2_5_VLForConditionalGeneration as QWenVLForConditionalGeneration
        else:
            from transformers import Qwen2VLForConditionalGeneration as QWenVLForConditionalGeneration

        self.model = QWenVLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype="auto", device_map="auto"
        )

    def _encode_image(self, image_path):
        pass
    
