import json
import re
import datetime, time
from pathlib import Path

class Timer:

    def __init__(self, log_path=None):
        self.log_path = log_path
        self.reset()

    def reset(self):
        self.t0 = time.time()
        self.tn = self.t0

    def step(self, title=''):
        t = time.time()
        message = f'{t-self.t0:8.2f} {t-self.tn:8.2f} {title}'
        self._log(message)
        self.tn = t

    def _log(self, message):
        if self.log_path:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with Path(self.log_path).open(mode='at') as fh:
                fh.write(f'{now} {message}\n')
        else:
            print(message)

def read_test_cases(path):
    if not path.exists():
        _error(f'Test cases file not found: {path}')
    return json.loads(path.read_text())

def get_image_paths(root_path, path_filter=None, test_cases=None) -> list[Path]:
    '''
    Returns a list of path to the input images.
    If test_cases is given, only return the paths that contain the case patterns (case-insensitive).
    If path_filter is given, only return the paths that contain that string (case-insensitive).
    If test_cases and path_filter are provided, only return their intersection.
    '''
    ret = list(Path(root_path).glob('**/*.jpg'))

    patterns_set = []
    if test_cases:
        patterns_set = [list(test_cases.keys())]
    if path_filter:
        patterns_set.append([path_filter])
    
    for patterns in patterns_set:
        patterns = '|'.join([
            re.escape(pattern)
            for pattern
            in patterns
        ])

        test_cases_regex = re.compile(patterns, re.IGNORECASE)

        def filter_path(path):
            path_with_size = f'{path}_{path.stat().st_size}'.lower()
            return test_cases_regex.search(path_with_size)

        ret = filter(
            filter_path,
            ret
        )
        ret = list(ret)

    return ret

def get_image_paths_sculpting(filter=None):
    # film_folder_names = (PARENT_PATH / 'settings/poc-films.json').read_text()
    # film_folder_names = json.loads(film_folder_names)

    ret = [
        path
        for path 
        in Path(IN_PATH).glob('**/shots/*/*.jpg')
        if (
            'trailer' not in str(path).lower()
            and re.findall(r'/videos/([^/]+)/', str(path))[0] in film_folder_names
            and (filter is None or filter in str(path).lower() or filter in str(path.stat().st_size))
        )
    ]

    return ret

def _error(message):
    print(f'ERROR: {message}')
    exit()
