from .base import ImageDescriber
from PIL import Image
from pathlib import Path
import datetime, time
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import re


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4"
MODEL_VERSION = ''
MAX_NEW_TOKENS = 512

class QwenVL(ImageDescriber):
    """Image description using Qwen2-VL model.

    https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4

    1.07B params, int4
    """    
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.cache = {
            'image_path': None,
            'image_encoding': None,
        }

    def get_name(self):
        return f'{MODEL_ID}:{MODEL_VERSION}'

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
        if not self.model:
            self._init_model()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
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
        inputs = inputs.to("cuda")

        #

        generated_ids = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        ret = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return ret[0]

    def _init_model(self):
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

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    def _new_model(self, use_cuda=False, use_attention=False):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype="auto", device_map="auto"
        )

    def _encode_image(self, image_path):
        pass
    
