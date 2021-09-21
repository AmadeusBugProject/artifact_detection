import glob
import json

import pandas
from sklearn import metrics
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.metrics import f1_score, roc_auc_score

from artifact_detection_model.constants import TARGET_NAMES
from artifact_detection_model.dataset_creation import get_training_and_test_set, get_manual_validation_data_set
from artifact_detection_model.regex_cleanup import regex_cleanup, is_markdown_artifact
from artifact_detection_model.utils.Logger import Logger
from artifact_detection_model.utils.paths import PROJECT_ROOT, REVIEWER_2_PATH, REVIEWER_1_PATH
from evaluation.artifact_detection_traning_learning_curve import validation_performance_on_dataset
from evaluation.utils import nlon_performance

log = Logger()

OUT_PATH = PROJECT_ROOT + 'evaluation/out/regex_classifier/'


class RegexPredict(LinearClassifierMixin):
    def predict(self, X):
        return [regex_predict(i) for i in X]


def regex_predict(line):
        artifacts, text = regex_cleanup([line])
        if is_markdown_artifact(line):
            return TARGET_NAMES['artifact']
        elif text:
            return TARGET_NAMES['text']
        else:
            return TARGET_NAMES['artifact']


def score_regex_line_classifier():
    pipeline = RegexPredict()

    df_train, df_test = get_training_and_test_set()
    data_validation = df_test.copy().pop('doc').values
    target_validation = df_test.copy().pop('target').values

    r1data, r1target = get_manual_validation_data_set(REVIEWER_1_PATH)
    r2data, r2target = get_manual_validation_data_set(REVIEWER_2_PATH)

    report = {'name': 'regex_classifier'}

    report.update(validation_performance_on_dataset(pipeline, data_validation, target_validation, 'test_set'))
    report.update(validation_performance_on_dataset(pipeline, r1data, r1target, 'reviewer_1'))
    report.update(validation_performance_on_dataset(pipeline, r2data, r2target, 'reviewer_2'))
    report.update(nlon_performance(pipeline, 'Fabio'))
    report.update(nlon_performance(pipeline, 'Mika'))
    report.update(nlon_performance(pipeline, 'agreement'))

    with open(OUT_PATH + 'regex_line_performance.txt', 'w') as fd:
        json.dump(report, fd, indent=2)
    return report, pipeline


def manual_validation_bug_classifier():
    bugs = get_bug_documents(REVIEWER_2_PATH)

    df = pandas.DataFrame()
    for url in list(bugs['url'].value_counts().index):
        target = bugs[bugs['url'] == url]['target'].to_list()
        doc = bugs[bugs['url'] == url]['doc'].to_list()
        document = '\n'.join(doc)
        predicted = pred_regex(document)
        bug_df = pandas.DataFrame({'target': target, 'predicted': predicted, 'doc': doc})
        df = df.append(bug_df)
    df = df[~df['target'].isnull()]
    df = df[~df['doc'].isnull()]

    y_predicted = df.pop('predicted').values
    target = df.pop('target').values
    name = 'complete_docs_regex_classifier'
    performance_report = {'man_validation_samples_' + name: len(y_predicted),
                          'classification_report_' + name: str(metrics.classification_report(target, y_predicted, target_names=TARGET_NAMES)),
                          'macro_f1_' + name: f1_score(target, y_predicted, average='macro'),
                          'roc-auc_' + name: roc_auc_score(target, y_predicted)}

    with open(OUT_PATH + 'regex_performance.txt', 'w') as fd:
        json.dump(performance_report, fd, indent=2)


def pred_regex(document):
    predicted = []
    inside_code_block = False
    for line in document.splitlines():
        if line.startswith('```'):
            inside_code_block = not inside_code_block
        if inside_code_block:
            predicted.append(TARGET_NAMES['artifact'])
        else:
            predicted.append(regex_predict(line))
    if document.endswith('\n'):
        predicted.append(1)
    return predicted


def get_bug_documents(path):
    df = pandas.read_csv(path, compression='zip')
    df['doc'] = df['doc'].fillna('')
    df['doc'] = df['doc'].astype(str)
    return df


def main():
    score_regex_line_classifier()
    manual_validation_bug_classifier()


if __name__ == "__main__":
    main()
