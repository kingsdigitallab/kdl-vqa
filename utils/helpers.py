import json
import re
import datetime, time
from pathlib import Path


DESCRIBE_PATH_IMAGES = 'test/data/images'
# PARENT_PATH = Path(__file__).parent
# IN_PATH = PARENT_PATH / '../data/rds2/STiC/PoC_2024/data/source/videos'

class Timer():

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

def get_image_paths(root_path, filter=None):
    ret = [
        path
        for path 
        in Path(root_path).glob('**/*.jpg')
        if (filter is None or filter in str(path).lower() or filter in str(path.stat().st_size))
    ]

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
