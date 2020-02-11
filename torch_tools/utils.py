from time import time
from tqdm import tqdm, tqdm_notebook
from .constants import VerbosityLevel


def numerical_order(files):
    return sorted(files, key=lambda x: int(x.split('.')[0]))


def in_jupyter():
    try:
        get_ipython()
        return True
    except Exception:
        return False


def make_verbose():
    if in_jupyter():
        return VerbosityLevel.JUPYTER
    else:
        return VerbosityLevel.CONSOLE


def wrap_with_tqdm(it, verbosity=make_verbose(), **kwargs):
    if verbosity == VerbosityLevel.SILENT or verbosity == False:
        return it
    elif verbosity == VerbosityLevel.JUPYTER:
        return tqdm_notebook(it, **kwargs)
    elif verbosity == VerbosityLevel.CONSOLE:
        return tqdm(it, **kwargs)


class Timer(object):
    def __init__(self):
        self._start = time()
        self._cumulative_time = 0.0
        self._resets_count = 0
        self._ignore_current = False

    def reset(self):
        current_time = time()
        diff = current_time - self._start
        self._start = current_time
        if not self._ignore_current:
            self._resets_count += 1
            self._cumulative_time += diff
        self._ignore_current = False
        return diff

    def avg(self):
        if self._resets_count > 0:
            return self._cumulative_time / self._resets_count
        else:
            return 0.0

    def ignore_current(self):
        self._ignore_current = True
