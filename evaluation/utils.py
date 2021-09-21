import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, f1_score, ConfusionMatrixDisplay

from artifact_detection_model.constants import TARGET_NAMES
from artifact_detection_model.dataset_creation import get_nlon_dataset
from artifact_detection_model.utils.paths import NLON_PATH, NLON_DATASETS


def nlon_performance(pipeline, reviewer):
    report = {}
    nlon_docs = []
    nlon_targets = []

    for name, csv_file in NLON_DATASETS:
        data, target = get_nlon_dataset(NLON_PATH + csv_file, reviewer)
        report.update(validation_performance_on_dataset(pipeline, data, target, name + '_' + reviewer))
        nlon_docs.extend(data)
        nlon_targets.extend(target)

    report.update(validation_performance_on_dataset(pipeline, nlon_docs, nlon_targets, 'nlon_all_' + reviewer))
    return report


def validation_performance_on_dataset(pipeline, data, target, name, output_misclassified=''):
    y_predicted = pipeline.predict(data)
    performance_report = {'man_validation_samples_' + name: len(data),
                          'classification_report_' + name: str(metrics.classification_report(target, y_predicted, target_names=TARGET_NAMES)),
                          'macro_f1_' + name: f1_score(target, y_predicted, average='macro'),
                          'roc-auc_' + name: roc_auc_score(target, y_predicted)}

    if output_misclassified:
        wrongly_identified_as_artifact = []
        wrongly_identified_as_text = []
        for index in range(0, len(data)):
            if target[index] == y_predicted[index]:
                pass
            elif target[index] == TARGET_NAMES['artifact'] and y_predicted[index] == TARGET_NAMES['text']:
                wrongly_identified_as_text.append(data[index])
            else:
                wrongly_identified_as_artifact.append(data[index])

        with open(output_misclassified + name + '_wrongly_identified_as_artifact.txt', 'w') as fd:
            fd.write('\n\n'.join(wrongly_identified_as_artifact))
        with open(output_misclassified + name + '_wrongly_identified_as_text.txt', 'w') as fd:
            fd.write('\n\n'.join(wrongly_identified_as_text))

    return performance_report


def plot_numpy_confusion_matrix(cm, target_names):
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=target_names)
    disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal', values_format=None)
    return disp
