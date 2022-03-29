import json
import time
import timeit

import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, f1_score, ConfusionMatrixDisplay

from artifact_detection_model.constants import TARGET_NAMES


def validation_performance_on_dataset(pipeline, data, target, name):
    num_runs_timeit = 10
    timeit_runtime = timeit.timeit(stmt='pipeline.predict(data_validation)', number=num_runs_timeit, globals={'pipeline': pipeline, 'data_validation': data}) / num_runs_timeit

    perf_start = time.perf_counter()
    y_predicted = pipeline.predict(data)
    perf_predict_runtime = time.perf_counter() - perf_start

    performance_report = {'man_validation_samples_' + name: len(data),
                          'classification_report_' + name: json.dumps(metrics.classification_report(target, y_predicted, target_names=TARGET_NAMES, output_dict=True)),
                          'macro_f1_' + name: f1_score(target, y_predicted, average='macro'),
                          'roc-auc_' + name: roc_auc_score(target, y_predicted),
                          'perf_predict_runtime_' + name: perf_predict_runtime,
                          'timeit_runtime_' + name: timeit_runtime}

    return performance_report


def plot_numpy_confusion_matrix(cm, target_names):
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=target_names)
    disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal', values_format=None)
    return disp


