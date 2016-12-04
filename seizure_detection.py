import json
import os.path
import numpy as np
from common import time
from common.data import CachedDataLoader, makedirs
from common.pipeline import Pipeline
from seizure.transforms import GetFeature
from seizure.tasks import TaskCore, MakePredictionsTask
from sklearn.linear_model import LogisticRegression

def run_seizure_detection(build_target):

    with open('SETTINGS.json') as f:
        settings = json.load(f)

    data_dir = str(settings['competition-data-dir'])
    cache_dir = str(settings['data-cache-dir'])
    submission_dir = str(settings['submission-dir'])
    figure_dir = str(settings['figure-dir'])

    makedirs(submission_dir)

    cached_data_loader = CachedDataLoader(cache_dir)
    splitsize = 30000
    ts = time.get_millis()
    bin_size = 50
    targets = [
            '1',
             '2',
             '3',
    ]
    pipelines = [
        # This is better than winning submission
        Pipeline(gen_preictal=True, pipeline=[GetFeature(50, 2500, 400, bin_size, 'usf',onlyfd_dfa=False,
                                                            with_dfa=False,with_dy=False,with_six=True,with_equal_freq=True,
                                                            with_mc=False,with_time_corr=True,smooth=True,smooth_Hz=160,power_edge=50,
                                                            with_square=True,with_log=False,with_sqrt=False,splitsize=splitsize,calibrate=False)]),
        #Pipeline(gen_preictal=True, pipeline=[only_FD_DFA(onlyfd_dfa=True)]),
    ]
    classifiers = [
        'GB',
        #'LSVC',
        # 'ET'

    ]
    cv_ratio = 0.5

    def should_normalize(classifier):
        clazzes = [LogisticRegression]
        return np.any(np.array([isinstance(classifier, clazz) for clazz in clazzes]) == True)

    def train_full_model(make_predictions):
        for pipeline in pipelines:
            for classifier in classifiers:
                print 'Using pipeline %s with classifier %s' % (pipeline.get_name(),classifier)
                guesses = ['File,Class']
                classifier_filenames = []
                #plot2file = PdfPages(os.path.join(figure_dir, ('figure%d-_%s_%s_.pdf' % (ts, classifier, pipeline.get_name()))))
                for target in targets:
                    task_core = TaskCore(cached_data_loader=cached_data_loader, data_dir=data_dir,
                                         target=target, pipeline=pipeline,
                                         classifier=classifier,
                                         normalize=should_normalize(classifier), gen_preictal=pipeline.gen_preictal,
                                         cv_ratio=cv_ratio,bin_size=bin_size)

                    if make_predictions:
                        predictions = MakePredictionsTask(task_core).run()
                        guesses.append(predictions.data)
                    else:
                        # task = TrainClassifierTask(task_core)
                        # task.run()
                        # classifier_filenames.append(task.filename())
                        print 'not implemented'

                if make_predictions:
                    filename = 'submission%d-%s_%s.csv' % (ts, classifier, pipeline.get_name())
                    filename = os.path.join(submission_dir, filename)
                    with open(filename, 'w') as f:
                        print >> f, '\n'.join(guesses)
                    print 'wrote', filename
                else:
                    print 'Trained classifiers ready in %s' % cache_dir
                    for filename in classifier_filenames:
                        print os.path.join(cache_dir, filename + '.pickle')


    if build_target == 'train_model':
        train_full_model(make_predictions=False)
    elif build_target == 'make_predictions':
        train_full_model(make_predictions=True)
    else:
        raise Exception("unknown build target %s" % build_target)
