# std lib
import re
import sys
import json
import argparse
import datetime, time
from pathlib import Path
import os
# this app
# from questions import questions
from describer.base import ImageDescriber
from utils.helpers import Timer, get_image_paths, _error, read_test_cases, get_repeat_ratio, format_answer

# CUDA_VISIBLE_DEVICES='' python [...] # to force CPU
# CUDA_VISIBLE_DEVICES=1 python [...] # to force 2nd GPU
# python describe.py --filter 99945 -q frame_accent -r
# nohup python describe.py --filter truman -r &
# # Specs below give comparable perfs to precision 3470 (cpu).
# srun -p cpu -c 16 --mem 4096 python describe.py -f 99945 -q frame_accent -r
# srun -p gpu -c 4 --mem 8192 --gpus-per-task 1 --constraint a100 -n 1 python describe.py -f 99945 -r
# srun -p interruptible_gpu -c 4 --mem 8192 --gpus-per-task 1 --constraint a40 -n 2 -t 2-0:00  python describe.py

# read BVQA_VENVS from environment variable
BVQA_VENVS = os.getenv('BVQA_VENVS', 'venvs')

VERSION = '0.3.0'
# TODO: use a 'dummy' model
DUMMY_DESCRIPTIONS = 0
DESCRIBE_IMAGE_LOCK_TIMEOUT_IN_SECONDS = 2 * 60
PATH_ROOT = 'data'
TARGET_FROM_TYPE = {
    'data': '.',
    'images': 'images',
    'questions': 'questions.json',
    'test_cases': 'test_cases.json',
    'answers': 'answers',
    'log': 'describe.log',
    'report': 'report.html',
}


class FrameQuestionAnswers:

    def __init__(self):
        self.reset()

    def reset(self, describer_name='moondream', model_id='', model_version='', filter='', max_images=0, redo=False, question_keys=None, optimise=False, root=PATH_ROOT, test=False):
        self.describer = None
        self.describer_name = describer_name
        self.model_id = model_id
        self.model_version = model_version
        self.root_path = Path(root)
        self.filter = filter
        self.max_images = max_images
        self.redo = redo
        self.question_keys = question_keys
        self.optimise = optimise
        self.test_cases = None
        if test: 
            self.test_cases = read_test_cases(self.get_path('test_cases'))
        self.timer = Timer(self.get_path('log'))
        self.get_path('answers').mkdir(parents=True, exist_ok=True)
    
    def new_describer(self):
        ret = ImageDescriber.new(self.describer_name, self.model_id, self.model_version)
        if ret is None:
            self._error(f'No describer supports the given parameters ({self.describer_name}, {self.model_id}, {self.model_version}).')
        ret.set_timer(self.timer)
        ret.set_optimisation(self.optimise)
        self.describer = ret
        return ret

    def get_path(self, type) -> Path:
        assert(type in TARGET_FROM_TYPE)
        return self.root_path / TARGET_FROM_TYPE[type]

    def process_command_line(self):
        actions = self._get_actions_info()
        epilog = 'Action:\n'
        for name, info in actions.items():
            epilog += f'  {name}:\n    {info["description"]}\n'

        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=epilog,
            description='Batch visual question answering.'
        )
        parser.add_argument("action", help="action to perform", choices=actions.keys())
        parser.add_argument('-d', '--describer', dest='describer_name', help='Name of the backend to describe the images (e.g. moondream).', default='moondream')
        parser.add_argument('-m', '--model', dest='model_id', help='ID of the huggingface model to describe the images (e.g. vikhyatk/moondream2).', default='')
        parser.add_argument('-v', '--version', dest='model_version', help='version/revision of the model, see model on hugging face (e.g. 2024-08-26).', default='')
        parser.add_argument('-f', '--filter', help='Filter image path by a string, e.g. batman')
        parser.add_argument('-q', '--questions', dest='question_keys', nargs='*', help='Only submit the questions with the given keys.')
        parser.add_argument('-o', '--optimise', action='store_true', help='Use model optimisations, if any (e.g. flash attention). Need higher specs.')
        parser.add_argument('-r', '--redo', action='store_true', help='Always submit questions again. Disregard cache.')
        parser.add_argument('--max-images', dest='max_images', type=int, default=0, help='Number of images to describe.')
        parser.add_argument('-R', '--root', help='Path to a data folder.', default=PATH_ROOT)
        parser.add_argument('-t', '--test', action='store_true', help='Process and test images listed in data/test_cases.json.')
        # todo
        # parser.add_argument('-s', '--settings', help='Path to a settings file.')
        # parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')
        self.args = parser.parse_args()

        self.reset(
            describer_name=self.args.describer_name,
            model_id=self.args.model_id,
            model_version=self.args.model_version,
            filter=self.args.filter,
            max_images=self.args.max_images,
            redo=self.args.redo,
            question_keys=self.args.question_keys,
            optimise=self.args.optimise,
            root=self.args.root,
            test=self.args.test
        )
        self.timer.step('=' * 40)
        command_line = ' '.join(sys.argv[1:])
        self.timer.step(f'args : {command_line}')

        action = actions.get(self.args.action, None)
        if action:
            action['method']()
        
        total_time = self.timer.get_time_since_reset()
        self.timer.step(f'DONE ({total_time:.2f} s.) - {command_line}')

    def _get_actions_info(self):
        ret = {}
        for k in dir(self):
            if k.startswith('action_'):
                name = k[7:]
                method = getattr(self, k)
                description = (method.__doc__ or '').split('\n')[0]
                ret[name] = {
                    'method': method,
                    'description': description
                }
        return ret

    def action_describe(self):
        '''Submit questions about multiple images to a visual model & save answers.'''
        self.new_describer()

        print(f'Describe with describer = {self.describer_name} ; model = {self.describer.get_name()}.')

        self.timer.step(f'model: {self.describer.get_name()}')
        import socket
        self.timer.step(f'host : {socket.gethostname()}')
        compute_info = self.describer.get_compute_info()
        self.timer.step(f'comp : {compute_info["desc"]}')
        self.timer.step(f'mem  : {(compute_info["size"] / 1024 / 1024 / 1024):.2f} GB')

        i = 0

        # third-party
        from tqdm import tqdm

        for image_path in (pbar := tqdm(self.get_image_paths())):
            qas_path = self.get_answer_path(image_path)

            res = self.describe_image(image_path)
            if res:
                i += 1
                if self.max_images and i >= self.max_images:
                    self.timer.step(f'INFO: stop after {self.max_images} images.')
                    break

        self.timer.step(f'described {i} images')

    def get_answer_path(self, image_path):
        return self.get_path('answers') / f'{image_path.name}_{image_path.stat().st_size}.qas.json'

    def get_image_paths(self):
        return get_image_paths(
            self.get_path('images'), 
            self.filter, 
            self.test_cases
        )

    def action_clear(self):
        '''Removes all answer files'''
        import shutil
        res = shutil.rmtree(str(self.get_path('answers')))

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

        qas_path = self.get_path('answers') / f'{image_path.name}_{image_path.stat().st_size}.qas.json'

        special_case = ''

        if qas_path.exists():
            ret = self.read_json_safe(qas_path)
            if ret is None:
                self.timer.step(f'WARNING: error reading {str(qas_path)}')
                return

        ret = self.upgrade_answers_format(ret, image_path)

        if DUMMY_DESCRIPTIONS:
            ret = None

        if ret:
            now = time.time()
            meta_started = ret['meta'].get('started', None)
            if meta_started and now - meta_started < DESCRIBE_IMAGE_LOCK_TIMEOUT_IN_SECONDS:
                special_case = 'image is already locked'
                ret = None
            else:
                questions_to_ask = {}
                questions = self.read_questions()
                if questions is None:
                    self._error('question file not found.')
                    
                for question_key, question in questions.items():
                    if self.question_keys and question_key not in self.question_keys:
                        continue
                    if question:
                        question_hash = self.get_hash_from_question(question)
                        if (self.redo 
                            or model_name not in ret['models'] 
                            or question_key not in ret['models'][model_name]['questions']
                            or ret['models'][model_name]['questions'][question_key]['hash'] != question_hash
                        ):
                            questions_to_ask[question_key] = question

                if questions_to_ask:
                    # lock the file
                    self.save_image_descriptions(qas_path, ret)

                    answers = self.describer.answer_questions(str(image_path), questions_to_ask)

                    if model_name not in ret['models']:
                        ret['models'][model_name] = {'questions': {}}
                    for question_key, answer in answers.items():
                        question = questions[question_key]
                        # question contains 'json' and the answer is valid json, convert to dict
                        if 'json' in question.lower():
                            inner_json = re.findall(r"('''|```)json\s*(\{.*\})\s*('''|```)", answer)
                            if inner_json:
                                answer = inner_json[0][1]
                            try:
                                answer = json.loads(answer)
                            except json.JSONDecodeError as e:
                                pass

                        answer_info = {
                            'answer': answer,
                            'hash': self.get_hash_from_question(question),
                        }
                        self.set_answer_correct(image_path, question_key, answer_info)
                        ret['models'][model_name]['questions'][question_key] = answer_info
                    # save and unlock
                    self.save_image_descriptions(qas_path, ret, True)
                else:
                    special_case = 'no unanswered question'
                    ret = None

        if special_case:
            special_case = ' - ' + special_case
        self.timer.step(f'describe image - after {special_case}')

        return ret

    def set_answer_correct(self, image_path, question_key, answer_info):
        '''
        Sets answer_info['correct'] = 0 or 1.
        1 if answer_info['answer'] matches the full list of conditions 
            stored in the test_cases file for image_path and question_key.
        0 otherwise.

        A condition is a python regular expression, e.g. "tree|plant".
        A negative condition starts with "-" (e.g. "-indoor").
        '''
        if not self.test_cases: return None

        # One test case looks like this:
        #
        # "susan": {
        #     "long_description": ["book|volumes"],
        #     "location": ["library", "-shop"],
        # },

        ret = False
        # TODO: avoid sequential search
        for pattern, question_conditions in self.test_cases.items():
            # TODO: also match against the file size
            if pattern.lower() in str(image_path).lower():
                conditions = question_conditions.get(question_key, [])
                if conditions:
                    answer = f"{answer_info['answer']}"
                    for condition in conditions:
                        regex = condition
                        if condition.startswith('-'):
                            regex = regex[1:]
                        ret = bool(re.search(regex, answer, re.IGNORECASE))
                        if condition.startswith('-'):
                            ret = not ret
                        if not ret: 
                            # print(f'incorrect: "{condition}" for "{answer}"')
                            break
            
                    answer_info['correct'] = int(ret)

        return ret

    def upgrade_answers_format(self, answers, image_path):
        ret = answers
            
        if ret:
            meta = ret.get('meta', {})
            if not meta:
                ret = None
            else:
                version = meta.get('version', '')
                if VERSION != version:
                    if version == '0.1.0':
                        # Upgarde structure
                        # in 0.2.0 answers go under 'models'
                        models = {}
                        for qk, q in ret['questions'].items():
                            if q['model'] not in models:
                                models[q['model']] = {'questions': {}}
                            models[q['model']]['questions'][qk] = q
                            del q['model']
                        ret['models'] = models
                        del ret['questions']
                        ret['meta']['version'] = '0.2.0'

                    if version == '0.2.0':
                        # added meta.image.path
                        ret['meta']['version'] = '0.3.0'

                    if ret['meta']['version'] != VERSION:
                        self._error(f'No upgrade path for answer format {version} -> {VERSION}.')

        if not ret:
            ret = {
                'meta': {
                    'started': 0,
                    'version': VERSION,
                },
                'models': {
                    # "vikhyatk/moondream2:2024-07-23": {
                    #     'questions': {
                    #         'frame_description': {
                    #             'answer': 'DUMMY_DESCRIPTIONS mode is ON',
                    #             'model': model_name,
                    #             'hash': hash,
                    #         }
                    #     }
                    # }
                },
            }

        image_info = ret['meta'].get('image', None)
        if not image_info:
            ret['meta']['image'] = {
                'path': str(Path(image_path).absolute().relative_to(self.root_path.absolute()))
            }

        return ret

    def read_questions(self):
        ret = None
        questions_path = self.get_path('questions')
        if questions_path.exists():
            ret = json.loads(questions_path.read_text())
        return ret
    
    def get_hash_from_question(self, question):
        # hashlib.md5()
        # base64.b64encode()
        return question

    def _error(self, message):
        self.timer.step(f'ERROR: {message}')
        _error(message)

    def action_load(self):
        '''Attempt to load the describer model.'''
        describer = self.new_describer()
        info = describer.get_compute_info()
        print(info)

    def action_build(self):
        '''Build a virtual environment with dependencies for the current describer.'''
        import os
        import subprocess
        import sys

        venv_path = os.path.join(BVQA_VENVS, self.describer_name)

        print(f'Rebuilding python environment {venv_path} ...')
        # requirements_base_txt = './requirements/base.txt'
        # requirements_describer_bash = f'./requirements/{self.describer_name}.bash'
        # requirements_describer_txt = f'./requirements/{self.describer_name}.txt'

        # delete virtual environment if it already exists
        if os.path.exists(venv_path):
            print(f'Remove existing environment.')
            import shutil
            shutil.rmtree(venv_path)

        # Create virtual environment
        if not os.path.exists(venv_path):
            print(f'Create environment.')
            subprocess.run([sys.executable, '-m', 'venv', venv_path], check=True)

        # Install base/common requirements, then the describer reqs
        requirements_stages = [
            'base',
            self.describer_name
        ]
        for requirements_stage in requirements_stages:
            # First try to run a custom bash script
            requirements_bash = f'./requirements/{requirements_stage}.bash'
            if os.path.exists(requirements_bash):
                venv_path_activate = os.path.join(venv_path, 'bin', 'activate')
                command = [
                    '/bin/bash',  
                    '-c',        
                    f'source "{venv_path_activate}" && cd "requirements" && source "{requirements_stage}.bash"'
                ]
                print(f'{requirements_stage}: {command}')
                try:
                    process = subprocess.run(command, text=True, check=True, capture_output=False)
                except subprocess.CalledProcessError as e:
                    print(f'FAIL: {e.stderr} [{requirements_stage}: {command}]')
                    exit()

                # print(f'Output: {process.stdout}')
                # print(f'Error: {process.stderr}')
            else:
                # Then install pip requirements for describer
                requirements_txt = f'./requirements/{requirements_stage}.txt'
                if os.path.exists(requirements_txt):
                    print(f'{requirements_stage}: {requirements_txt}')
                    subprocess.run([f'{venv_path}/bin/pip', 'install', '-r', requirements_txt], check=True)

        # Install base dependencies
        # subprocess.run([f'{venv_path}/bin/pip', 'install', '-r', requirements_base_txt], check=True)
        # subprocess.run([f'{venv_path}/bin/ltt', 'install', 'torch', 'torchvision', 'accelerate'], check=True)
        # subprocess.run([f'{venv_path}/bin/pip-sync', requirements_describer_txt], check=True)
        # subprocess.run([f'{venv_path}/bin/pip', 'install', '-r', requirements_describer_txt], check=True)

        print(f'\nVirtual environment created at {venv_path} with dependencies installed.')
        print('Use the following command to activate the virtual environment:')
        print(f'\nsource {venv_path}/bin/activate')

    def action_report(self):
        '''Generate an HTML report from the answer files.'''
        # TODO: Need to refactor this very ugly code.
        # Use template or and externalise to improve readability.
        report_path = self.get_path('report')
        data_path = self.get_path('data')

        questions = self.read_questions()
        questions_html = ''.join([f'<li><strong>{k}</strong>: {v}</li>' for k,v in questions.items()])
        questions_html = f'<ul>{questions_html}</ul>'

        # TODO: summary = table with accuracy % for each model vs question.
        # Could also summarise the time. See gh-10
        summary = ''
        stats = {}
        images = ''
        i = 0
        for image_path in self.get_image_paths():
            i += 1
            images += '<div>'
            images += f'<h3>{i}. {image_path.relative_to(data_path)}</h3>'
            image_relative_path = image_path.relative_to(report_path.parent)
            images += f'<a href="{image_relative_path}"><img src="{image_relative_path}"></a>'
            answers_path = self.get_answer_path(image_path)

            questions = self.read_questions()

            if answers_path.exists():
                answers = json.loads(answers_path.read_text())
                for model_id, model_info in answers.get('models', {}).items():
                    if self.model_id and self.model_id not in model_id: continue
                    images += f'<h4>{model_id}</h4>'
                    images += '<ul>'
                    for question_key, question_info in model_info['questions'].items():
                        if question_key not in questions: continue

                        if self.question_keys and question_key not in self.question_keys: continue

                        correctness = ''
                        is_correct = question_info.get('correct', None)
                        repeat_ratio = get_repeat_ratio(question_info["answer"])
                        if is_correct == 0:
                            correctness = '<span class="incorrect">[WRONG]</span>'
                        if is_correct == 1:
                            correctness = '<span class="correct">[RIGHT]</span>'
                        if repeat_ratio > 0.5:
                            correctness += f'<span class="repeat">[REPEAT]</span>'
                            if is_correct is None:
                                is_correct = 0
                        answer_html = format_answer(question_info['answer'])
                        images += f'<li><span class="question-key">{question_key}</span>: {correctness} {answer_html}</li>'

                        if model_id not in stats:
                            stats[model_id] = {}
                        if question_key not in stats[model_id]:
                            stats[model_id][question_key] = {'correct': 0, 'total': 0, 'repeat': 0}
                        if is_correct is not None:
                            stats[model_id][question_key]['total'] += 1
                        if is_correct == 1:
                            stats[model_id][question_key]['correct'] += 1
                        if repeat_ratio > 0.5:
                            stats[model_id][question_key]['repeat'] += 1

                    images += '</ul>'

            images += '</div>'

        header = ''
        for question_key in questions:
            header += f'<th>{question_key}</th>'
        header = f'<tr><th>model</th>{header}</tr>'

        for model_id, model_info in stats.items():
            row = f'<td>{model_id[:30]}</td>'
            for question_key in questions:
            # for question_key, question_info in model_info.items():
                question_info = model_info.get(question_key, None)
                if question_info and question_info['total']:
                    accuracy = 0
                    if question_info['total']:
                        accuracy = question_info['correct'] / question_info['total']
                    row += f'<td>{accuracy * 100:.1f}% ({question_info["correct"]} / {question_info["total"]})</td>'
                else:
                    row += f'<td></td>'
            summary += f'<tr>{row}</tr>'
        
        summary = f'<table>{header}{summary}</table>'

        # TODO: improve HTML format, and move template to external file
        content = ('''
        <html>
        <head>
            <title>Report - BVQA</title>
            <style>
            img {
                height: 10em;
            }
            .question-key {
                font-weight: bold;
            }
            .correct {
                background-color: lightgreen;
            }
            .incorrect {
                background-color: pink;
            }
            .repeat {
                background-color: orange;
            }
            table, th, td {
                border: 1px solid black;
                border-collapse: collapse;
                padding: 0.2em;
            }
            .format-json {
                font-family: "Lucida Console", Monaco, monospace;
                display: block;
                font-size: 0.8em;
            }
            </style>
        </head>
        <body>
            <h2>Questions</h2>''' +
        questions_html +
        '''<h2>Summary</h2>''' +
        summary +
        '''<h2>Images</h2>''' +
        images +
        '''
        </body>
        </html>''')
        report_path.write_text(content)

        print(f'WRITTEN {report_path}')


if __name__ == '__main__':
    # Code to execute only when run as a script
    fqas = FrameQuestionAnswers()
    fqas.process_command_line()
