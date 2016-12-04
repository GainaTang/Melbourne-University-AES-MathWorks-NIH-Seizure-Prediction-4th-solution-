from collections import namedtuple
#from common.data import CachedDataLoader, makedirs
import os.path
from common import io
import os



TaskCore = namedtuple('TaskCore', ['cached_data_loader', 'data_dir', 'target', 'pipeline',
                                   'classifier', 'normalize', 'gen_preictal', 'cv_ratio','bin_size'])

class jsdict(dict):
    def __init__(self, data):
        self.__dict__ = data

def load(filename):
    def wrap_data(data):
        if isinstance(data, list):
            return [jsdict(x) for x in data]
        else:
            return jsdict(data)

    if filename is not None:
        filename = os.path.join('data-cache',filename)
        data = io.load_hkl_file(filename)
        if data is not None:
            return wrap_data(data)
    return wrap_data(data)

def TaskCom():
    guesses = ['File,Class']
    f1='predictions_1_gen_shanno--resampled_100Hz-six-dy-mobility_complexsity-time_corr-square-power_edge-50_GB'
    f2='predictions_2_gen_shanno--resampled_100Hz-dy-mobility_complexsity-time_corr-square-power_edge-50_GB'
    f3='predictions_3_gen_shanno--resampled_100Hz-time_corr-equal_freq-power_edge-50_GB'
    guesses.append(load(f1).data)
    guesses.append(load(f2).data)
    guesses.append(load(f3).data)
    submission_dir = 'submissions'
    filename = 'submission_com-%s+%s+%s.csv' % (f1[26:],f2[26:],f3[26:])
    filename = os.path.join(submission_dir, filename)
    with open(filename, 'w') as f:
        print >> f, '\n'.join(guesses)
    print 'wrote', filename

if __name__ == "__main__":
    TaskCom()

