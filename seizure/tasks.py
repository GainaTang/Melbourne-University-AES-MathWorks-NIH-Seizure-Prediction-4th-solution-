from collections import namedtuple
import hickle as hkl
import os.path
import numpy as np
import scipy.io
import common.time as time
from sklearn import preprocessing
import graphlab
from sklearn.model_selection import GroupKFold, StratifiedKFold, GridSearchCV, cross_val_score, RandomizedSearchCV,PredefinedSplit
import random
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LassoCV,LogisticRegression
from sklearn.feature_selection import RFECV
from scipy.stats.mstats import *


TaskCore = namedtuple('TaskCore', ['cached_data_loader', 'data_dir', 'target', 'pipeline',
                                   'classifier', 'normalize', 'gen_preictal', 'cv_ratio','bin_size'])


class Task(object):

    def __init__(self, task_core):
        self.task_core = task_core

    def filename(self):
        raise NotImplementedError("Implement this")

    def run(self):
        return self.task_core.cached_data_loader.load(self.filename(), self.load_data)

class Loadpreictal_FD_DFA_DataTask(Task):
    """
    Load the preictal mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X, 'Y': y, 'latencies': latencies}
    """

    def filename(self):
        return 'data_preictal_FD_DFA_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        return parse_input_data(self.task_core.data_dir, self.task_core.target, '1', self.task_core.pipeline,
                                self.task_core.gen_preictal)
class Loadinterictal_FD_DFA_DataTask(Task):
    """
    Load the preictal mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X, 'Y': y, 'latencies': latencies}
    """

    def filename(self):
        return 'data_interictal_FD_DFA_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        return parse_input_data(self.task_core.data_dir, self.task_core.target, '0', self.task_core.pipeline,
                                self.task_core.gen_preictal)

class LoadTest_FD_DFA_DataTask(Task):
    """
    Load the test mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X}
    """

    def filename(self):
        return 'data_test_FD_DFA_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        return parse_input_data_test(self.task_core.data_dir, self.task_core.target, self.task_core.pipeline,
                                     self.task_core.bin_size)


class LoadpreictalDataTask(Task):
    """
    Load the preictal mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X, 'Y': y, 'latencies': latencies}
    """

    def filename(self):
        return 'data_preictal_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        return parse_input_data(self.task_core.data_dir, self.task_core.target, '1', self.task_core.pipeline,
                                self.task_core.gen_preictal)


class LoadInterictalDataTask(Task):
    """
    Load the interictal mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X, 'Y': y}
    """

    def filename(self):
        return 'data_interictal_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        return parse_input_data(self.task_core.data_dir, self.task_core.target, '0', self.task_core.pipeline)


class LoadTestDataTask(Task):
    """
    Load the test mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X}
    """

    def filename(self):
        return 'data_test_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        return parse_input_data_test(self.task_core.data_dir, self.task_core.target, self.task_core.pipeline,self.task_core.bin_size)

class MakePredictionsTask(Task):
    """
    Make predictions on the test data.
    """

    def filename(self):
        return 'predictions_%s_%s_%s' % (
        self.task_core.target, self.task_core.pipeline.get_name(), self.task_core.classifier)

    def load_data(self):
        preictal_data = LoadpreictalDataTask(self.task_core).run()
        interictal_data = LoadInterictalDataTask(self.task_core).run()
        test_data = LoadTestDataTask(self.task_core).run()
        X_test = test_data.X
        test_SPF = test_data.SamplePerFile
        X = np.concatenate((preictal_data.X, interictal_data.X), axis=0)
        y = np.concatenate((np.ones(preictal_data.y.shape), interictal_data.y), axis=0)
        h_num = np.concatenate((preictal_data.h_num, interictal_data.h_num))
        testfold = stratified_group_kfold(X, y, h_num, 5)
        ps = PredefinedSplit(testfold)

        if self.task_core.classifier == 'GB':

            trainCLF = GradientBoostingClassifier(n_estimators=6000, max_depth=10,min_samples_leaf=5,
                                           min_samples_split=2, learning_rate=0.001,
                                           max_features=40, subsample=0.65)
        elif self.task_core.classifier == 'voting':
            clf1 = LogisticRegression(random_state=1)
            clf2 = RandomForestClassifier(random_state=1)
            clf3 = GaussianNB()
            eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
            params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200],}
            print params
            trainCLF = GridSearchCV(estimator=eclf, param_grid=params, cv=ps)
        elif self.task_core.classifier == 'lasso':
            param_grid={
                'n_alphas':[200,400,600],
                'max_iter':[2000,4000]
            }
            trainCLF=LassoCV(max_iter=2000, cv=ps, n_jobs=-1,n_alphas=200)
        elif self.task_core.classifier == 'GB_fl':
            trainCLF = RFECV(GradientBoostingClassifier(n_estimators=100, min_samples_leaf=5,
                                                        max_depth=15,
                                                        min_samples_split=2, learning_rate=0.2,
                                                        max_features='sqrt', subsample=0.65,
                                                        random_state=0, ),n_jobs=-1, cv=ps, scoring='roc_auc')
        elif self.task_core.classifier == 'NB':
            trainCLF = GaussianNB()
        if self.task_core.pipeline.transforms[0].calibrate:
            trainCLF = CalibratedClassifierCV(GradientBoostingClassifier(max_depth=10,min_samples_leaf=5,
                                                                             min_samples_split=2,
                                                                             n_estimators=6000,
                                                                             learning_rate=0.001,max_features=40,
                                                                             subsample=0.65), method='sigmoid', cv=ps)
        trainCLF.fit(X, y)
        predicted = trainCLF.predict_proba(X_test)
        SamplePerFile = test_SPF

        return make_predictions(self.task_core.target, predicted, SamplePerFile, len(SamplePerFile))

def load_mat_data(data_dir, target, component, labels):
    dir = os.path.join(data_dir, 'train_' + target)
    i = 0
    im = labels['image']
    safe = labels['safe']
    clas = labels['class']
    for j in range(len(im)):
        name = im[j]
        if name[0] == target and safe[j] == 1 and clas[j] == int(component):
            filename = '%s/%s' % (dir, name)
            if os.path.exists(filename):
                try:
                    data = scipy.io.loadmat(filename)
                except:
                    print 'data corruption, skipping'
                    data['dataStruct'][0][0][0] = np.zeros((240000, 16))
                    data['dataStruct'][0][0][4][0][0] += 1
                    yield (data)
                else:
                    yield (data)
            else:
                if i == 1:
                    raise Exception("file %s not found" % filename)
def parse_input_data(data_dir, target, data_type, pipeline):
    preictal = data_type == '1'
    interictal = data_type == '0'
    labels = graphlab.SFrame.read_csv('seizure-data/train_and_test_data_labels_safe.csv')
    mat_data = load_mat_data(data_dir, target, data_type, labels)

    def process_raw_data(mat_data,splitsize):
        start = time.get_seconds()
        print 'Loading data',
        X = []
        y = []
        h_num = []
        cc = 0
        hour_num = 0
        pre_sequence_num = 0
        for segment in mat_data:
            cc += 1
            print cc
            for skey in segment.keys():
                if "data" in skey.lower():
                    mykey = skey
            try:
                sequence_num = segment[mykey][0][0][4][0][0]
            except:
                sequence_num = random.randint(1, 6)
            print 'seq: %d' % (sequence_num)
            if sequence_num == pre_sequence_num + 1:
                hour_num = hour_num
            else:
                hour_num += 1
            print "hour_num: %d" % (hour_num)
            pre_sequence_num = sequence_num
            if preictal:
                try:
                    preictual_sequence = segment[mykey][0][0][4][0][0]
                except:
                    preictual_sequence = 1
                else:
                    pass
                y_value = preictual_sequence  # temporarily set to sequence number
            elif interictal:
                y_value = 0

            data = segment[mykey][0][0][0]
            # if target == '2':
            #     data = np.delete(data, [3, 9], 1)
            data_tmp = data[np.invert(np.all(data==0, axis=1))]
            if data_tmp.shape[0]<=2000:
                 print 'too much zeros, skipping'
                 continue
            sampleSizeinSecond = data_tmp.shape[0] / 400
            data = data_tmp.transpose()
            axis = data_tmp.ndim - 1
            # tic=time.get_seconds()
            print sampleSizeinSecond
            '''DataSampleSize: split the 10 minutes data into several clips:
            For one second data clip, patient1 and patient2 were finished in 3 hours. Dog1 clashed after 7+ hours for out of memory
            try ten second data clip
            '''
            DataSampleSize = splitsize  # data.shape[1]/(totalSample *1.0)  #try to split data into equal size
            splitIdx = np.arange(DataSampleSize, data.shape[1], DataSampleSize)
            splitIdx = np.int32(np.ceil(splitIdx))
            splitData = np.hsplit(data, splitIdx)
            SPF = 0
            for s in splitData:
                if s.shape[1] < 5000:  #is not so sparse
                    continue

                else:
                    transformed_data = pipeline.apply(s)
                    X.append(transformed_data)
                    y.append(y_value)
                    h_num.append(hour_num)
                    SPF += 1
                    if np.any(np.isnan(transformed_data)) or np.any(np.isinf(transformed_data)):
                        print 'bug'
            print 'done'

        print '(%ds)' % (time.get_seconds() - start)

        X = np.array(X)
        y = np.array(y)
        h_num = np.array(h_num)
        print 'X', X.shape, 'y', y.shape
        return X, y, h_num

    splitsize= pipeline.transforms[0].splitsize
    data = process_raw_data(mat_data,splitsize)
    X, y, h_num = data
    if interictal:
        h_num += 200
    return {
        'X': X,
        'y': y,
        'h_num': h_num
    }


def load_mat_data_test(data_dir, target):
    dir = os.path.join(data_dir, 'test_' + target + '_new')
    done = False
    i = 0
    while not done:
        i += 1
        nstr = '%d' % i

        filename = '%s/new_%s_%s.mat' % (dir, target, nstr,)
        if os.path.exists(filename):
            try:
                data = scipy.io.loadmat(filename)
            except:
                print 'data corruption, skipping'
            else:
                yield (data)
        else:
            if i == 1:
                raise Exception("file %s not found" % filename)
            done = True


def parse_input_data_test(data_dir, target, pipeline):

    mat_data = load_mat_data_test(data_dir, target)
    def process_raw_data(mat_data,splitsize):
        start = time.get_seconds()
        print 'Loading data',
        # print mat_data
        SamplePerFile = []
        X = []
        y = []
        cc = 0
        for segment in mat_data:
            cc += 1
            print cc
            for skey in segment.keys():
                if "data" in skey.lower():
                    mykey = skey
            data = segment[mykey][0][0][0]
            if np.all(data == 0):
                print 'All of data zero, filling random numbers'
                for s in range(int(240000/splitsize)):
                    transformed_data = np.random.randn(transformed_data_length)
                    X.append(transformed_data)
                SamplePerFile.append(int(240000/splitsize))
                continue
            data_tmp = data[np.invert(np.all(data == 0, axis=1))]
            sampleSizeinSecond = data_tmp.shape[0] / 400
            data = data_tmp.transpose()
            axis = data.ndim - 1

            print sampleSizeinSecond

            '''DataSampleSize: split the 10 minutes data into several clips:
            For one second data clip, patient1 and patient2 were finished in 3 hours. Dog1 clashed after 7+ hours for out of memory
            try ten second data clip
            '''
            DataSampleSize = splitsize  # data.shape[1] / (totalSample * 1.0)  # try to split data into equal size
            splitIdx = np.arange(DataSampleSize, data.shape[1], DataSampleSize)
            splitIdx = np.int32(np.ceil(splitIdx))
            splitData = np.hsplit(data, splitIdx)
            SPF = 0
            #pre_sample_size = 0
            #channel = 16
            # if target == '2':
            #     channel = 14
            for s in splitData:
                transformed_data = pipeline.apply(s)
                X.append(transformed_data)
                SPF += 1
            SamplePerFile.append(SPF)
            print 'done'
            transformed_data_length=transformed_data.shape[0]
        X = np.array(X)
        print 'X', X.shape
        return X, SamplePerFile
    splitsize=pipeline.transforms[0].splitsize
    data, SamplePerFile = process_raw_data(mat_data,splitsize)
    return {
        'X': data,
        'SamplePerFile': SamplePerFile
    }
def flatten(data):
    if data.ndim > 2:
        return data.reshape((data.shape[0], np.product(data.shape[1:])))
    else:
        return data

def translate_prediction(prediction):
    if prediction.shape[0] == 7:
        interictal, p1, p2, p3, p4, p5, p6 = prediction
        preictal = p1 + p2 + p3 + p4 + p5 + p6

        return preictal
    elif prediction.shape[0] == 2:
        interictal, p1 = prediction
        preictal = p1
        return preictal
    elif prediction.shape[0] == 1:
        return prediction[0]
    else:
        raise NotImplementedError()
def make_predictions(target, predictions, SamplePerFile, numFile):
    lines = []
    cumSample = 0
    for i in range(numFile):
        j = i + 1
        nstr = '%d' % j

        preictal_segments = []
        for k in range(SamplePerFile[i]):
            p = predictions[cumSample + k]
            preictal = translate_prediction(p)
            preictal_segments.append(preictal)
        cumSample += SamplePerFile[i]
        preictalOverAllSample = get_combine_prediction(preictal_segments)
        lines.append('new_%s_%s.mat,%.15f' % (target, nstr, preictalOverAllSample))

    return {
        'data': '\n'.join(lines)
    }


def get_combine_prediction(preictal_segments):

    interictal_amean = 1.0 - np.mean(preictal_segments)
    interictal = 1.0 - np.array(preictal_segments)
    if np.any(interictal == 0):
        interictal_gmean = interictal_amean
        interictal_hmean = interictal_amean
    else:
        interictal_gmean = gmean(interictal)
        interictal_hmean = hmean(interictal)
    interictal_agmean = 0.5 * (interictal_amean + interictal_gmean)
    interictal_hgmean = 0.5 * (interictal_hmean + interictal_gmean)
    return 1.0 - interictal_hmean
def stratified_group_kfold(y,group,K):
    testfold=np.zeros(y.shape[0])
    zero_pool = np.asarray(np.where(y == 0)).flatten()
    one_pool = np.asarray(np.where(y == 1)).flatten()
    for kk in range(K):
        zero_target = zero_pool.shape[0]/(K-kk)
        one_target = one_pool.shape[0]/(K-kk)

        test_zero_pool = np.random.choice(zero_pool,size=zero_target)
        test_zero_index = []
        test_one_pool = np.random.choice(one_pool,size=one_target)
        test_one_index = []
        for i in test_zero_pool:
            if len(test_zero_index)<= zero_target:
                tmp = np.array(np.where(group==group[i])).ravel()
                for j in tmp:
                    test_zero_index.append(j)
        for i in test_one_pool:
            if len(test_one_index)<= one_target:
                tmp = np.array(np.where(group==group[i])).ravel()
                for j in tmp:
                    test_one_index.append(j)
        test_zero_index = np.unique(test_zero_index)
        test_one_index = np.unique(test_one_index)
        test_index = np.concatenate((test_one_index,test_zero_index))
        zero_pool = np.setdiff1d(zero_pool, test_zero_index)
        one_pool = np.setdiff1d(one_pool, test_one_index)
        testfold[test_index]=kk
    return testfold




