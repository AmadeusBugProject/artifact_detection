import pickle
import sys
import time
import timeit

import joblib
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, confusion_matrix, plot_confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from artifact_detection_model.SpecialCharacterToWords import SpecialCharacterToWords
from artifact_detection_model.constants import TARGET_NAMES
from artifact_detection_model.utils.Logger import Logger

log = Logger()


def run_ml_artifact_training(df_train, clf):
    data_train = df_train.copy().pop('doc').values
    target_train = df_train.copy().pop('target').values

    pipeline = Pipeline([
        ('charrep', SpecialCharacterToWords()),
        ('vect', CountVectorizer()),
        ('clf', clf)])

    parameters = {
        'charrep__repl_all_caps': False,
        'vect__ngram_range': (1, 3),
        'vect__stop_words': None,
        'vect__lowercase': False,
    }

    log.s("train_samples: %d" % len(data_train))

    perf_start = time.perf_counter()

    pipeline.set_params(**parameters)
    pipeline.fit(data_train, target_train)

    perf_train_runtime = time.perf_counter() - perf_start

    pipeline_pickle = pickle.dumps(pipeline)
    model_size = sys.getsizeof(pipeline_pickle)/(1000.*1024.)

    performance_report = {'train_samples': len(data_train),
                          'params': str(parameters),
                          'perf_train_runtime': perf_train_runtime,
                          'model_size': model_size}

    return performance_report, pipeline