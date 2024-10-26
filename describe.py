# std lib
import sys
import json
import argparse
import datetime, time
from pathlib import Path
# this app
from questions import questions
from describer.base import ImageDescriber
from utils.helpers import Timer, get_image_paths
# third-party
from tqdm import tqdm

# CUDA_VISIBLE_DEVICES='' python [...] # to force CPU
# CUDA_VISIBLE_DEVICES=1 python [...] # to force 2nd GPU
# python describe.py --filter 99945 -q frame_accent -r
# nohup python describe.py --filter truman -r &
# # Specs below give comparable perfs to precision 3470 (cpu).
# srun -p cpu -c 16 --mem 4096 python describe.py -f 99945 -q frame_accent -r
# srun -p gpu -c 4 --mem 8192 --gpus-per-task 1 --constraint a100 -n 1 python describe.py -f 99945 -r
# srun -p interruptible_gpu -c 4 --mem 8192 --gpus-per-task 1 --constraint a40 -n 2 -t 2-0:00  python describe.py

VERSION = '0.1.0'
# TODO: use a 'dummy' model
DUMMY_DESCRIPTIONS = 0

DESCRIBE_PATH_IMAGES = '../../../data/rds2/STiC/PoC_2024/data/source/videos'
DESCRIBE_PATH_ANSWERS = '../../../data/etl/descriptions'
DESCRIBE_PATH_LOG = f'{DESCRIBE_PATH_ANSWERS.rstrip('/')}/describe.log'
DESCRIBE_IMAGE_LOCK_TIMEOUT_IN_SECONDS = 2 * 60

class FrameQuestionAnswers():

    def __init__(self):
        self.reset()

    def reset(self, model_name='moondream', filter='', max_images=0, redo=False, question_keys=None, optimise=False):

        self.filter = filter
        self.max_images = max_images
        self.redo = redo
        self.question_keys = question_keys
        self.optimise = optimise
        self.describer = ImageDescriber.new(model_name)
        self.describer.set_optimisation(self.optimise)
        self.timer = Timer(LOG_PATH)
        self.describer.set_timer(self.timer)

        Path(DESCRIBE_PATH_ANSWERS).mkdir(parents=True, exist_ok=True)

    def process_command_line(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', '--filter', help='Filter image path by a string, e.g. batman')
        parser.add_argument('-m', '--model', help='Name of the model to describe the images.', default='moondream')
        parser.add_argument('-q', '--questions', nargs='*', help='Only submit the questions with the given keys.')
        parser.add_argument('-o', '--optimise', action='store_true', help='Use model optimisations, if any (e.g. flash attention). Need higher specs.')
        parser.add_argument('-r', '--redo', action='store_true', help='Always submit questions again. Disregard cache.')
        parser.add_argument('--max-images', dest='max_images', type=int, default=0, help='Number of images to describe.')
        parser.add_argument('-R', '--root', help='Path to a data folder.')
        parser.add_argument('-s', '--settings', help='Path to a settings file.')
        parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')
        self.args = parser.parse_args()

        self.reset(
            model_name=self.args.model,
            filter=self.args.filter,
            max_images=self.args.max_images,
            redo=self.args.redo,
            question_keys=self.args.questions,
            optimise=self.args.optimise,
        )

        self.describe_images()

    def describe_images(self):
        self.timer.step('=' * 40)
        self.timer.step(f'-m {self.describer.get_name()} -f {self.filter}')

        i = 0

        image_paths = get_image_paths(self.config['PATH_IMAGES'], self.filter)

        for image_path in (pbar := tqdm(image_paths)):
            qas_path = Path(DESCRIBE_PATH_ANSWERS) / f'{image_path.name}_{image_path.stat().st_size}.qas.json'

            # pbar.set_postfix_str(qas_path.name)

            res = self.describe_image(image_path)
            if res:
                i += 1
                if self.max_images and i >= self.max_images:
                    print(f'INFO: stop after {self.max_images} images.')
                    break

        self.timer.step('DONE - described images')


    def save_image_descriptions(self, path, descriptions, unlock=False):
        descriptions['meta']['started'] = 0 if unlock else time.time() 
        path.write_text(json.dumps(descriptions, indent=2))


    def read_json_safe(self, path):
        """Read an existing json file.
        Return content as a dict.
        If the file is not valid json, sleep 3s then read again.
        If still not valid json, we return None.
        This deals with case where concurrent process 
        was in the middle of writing the file.
        """
        ret = None
        for i in [0, 1]:
            try:
                ret = json.loads(path.read_text())
                break
            except Exception as e:
                self.timer.step(f'WARNING: error reading {str(path)} - {str(e)}')
                if i == 0:
                    time.sleep(3)
        return ret
        
    def describe_image(self, image_path):
        self.timer.step(f'describe image - before - {image_path}')
        model_name = self.describer.get_name()
        ret = None

        qas_path = Path(DESCRIBE_PATH_ANSWERS) / f'{image_path.name}_{image_path.stat().st_size}.qas.json'

        if qas_path.exists():
            ret = self.read_json_safe(qas_path)
            if ret is None:
                self.timer.step(f'WARNING: error reading {str(qas_path)}')
                return

            meta = ret.get('meta', {})
            if not meta:
                ret = None
            else:
                version = meta.get('version', '')
                # TODO: only for breaking changes
                if VERSION != version:
                    ret = None
        
        if not ret:
            ret = {
                'meta': {
                    'started': 0,
                    'version': VERSION,
                },
                'questions': {
                    # 'frame_description': {
                    #     'answer': 'DUMMY_DESCRIPTIONS mode is ON',
                    #     'model': model_name,
                    #     'hash': hash,
                    # }
                }
            }

        if DUMMY_DESCRIPTIONS:
            ret = None

        if ret:
            now = time.time()
            if now - ret['meta']['started'] < DESCRIBE_IMAGE_LOCK_TIMEOUT_IN_SECONDS:
                print('WARNING: image is already locked.')
            else:
                questions_to_ask = {}
                for question_key in questions.keys():
                    if self.question_keys and question_key not in self.question_keys:
                        continue
                    question = questions[question_key]
                    if question:
                        question_hash = self.get_hash_from_question(question)
                        if self.redo or question_key not in ret['questions'] or ret['questions'][question_key]['hash'] != question_hash:
                            questions_to_ask[question_key] = question
                            # self.timer.step(f'question - before - {question_key}')
                            # ret['questions'][question_key] = {
                            #     'answer': self.describer.answer_question(str(image_path), question),
                            #     'model': model_name,
                            #     'hash': question_hash
                            # }
                            # self.save_image_descriptions(qas_path, ret)

                if questions_to_ask:
                    # lock the file
                    self.save_image_descriptions(qas_path, ret)
                    answers = self.describer.answer_questions(str(image_path), questions_to_ask)
                    for question_key, answer in answers.items():
                        ret['questions'][question_key] = {
                            'answer': answer,
                            'model': model_name,
                            'hash': self.get_hash_from_question(questions[question_key])
                        }

                    # save and unlock
                    self.save_image_descriptions(qas_path, ret, True)

        self.timer.step(f'describe image - after')

        return ret

    def get_hash_from_question(self, question):
        # hashlib.md5()
        # base64.b64encode()
        return question

if __name__ == '__main__':
    # Code to execute only when run as a script
    fqas = FrameQuestionAnswers()
    fqas.process_command_line()
