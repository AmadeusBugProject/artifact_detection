import json

import joblib
import pandas
from sklearn.svm import LinearSVC

from artifact_detection_model.constants import TARGET_NAMES
from artifact_detection_model.model_training import run_ml_artifact_training
from artifact_detection_model.utils.Logger import Logger
from datasets.constants import LANGUAGES
from datasets.dataset_utils import get_trainingset, get_all_validation_sets
from evaluation.utils import validation_performance_on_dataset
from file_anchor import root_dir

log = Logger()

OUT_PATH = root_dir() + 'artifact_detection_model/out/'


def main():
    for lang in LANGUAGES:
        train_language(lang)
    train_multi_language()


def train_multi_language():
    seed = 42
    val_sets = get_all_validation_sets()
    train_size = 200000

    df_sel = pandas.DataFrame()
    for lang in LANGUAGES:
        df_train = get_trainingset(lang, balance=False)
        df_sel = df_sel.append(df_train[df_train['target'] == 1].sample(
            int(train_size / (2 * len(LANGUAGES))), random_state=seed, replace=True))
        df_sel = df_sel.append(df_train[df_train['target'] == 0].sample(
            int(train_size / (2 * len(LANGUAGES))), random_state=seed, replace=True))

    report, pipeline = run_ml_artifact_training(df_sel, LinearSVC(random_state=42))

    report.update({'seed': seed})
    report.update({'train_frac': train_size})

    for val_set_name, val_set_df in val_sets.items():
        val_docs = val_set_df.copy().pop('doc').values
        val_targets = val_set_df.copy().pop('target').values
        report.update(validation_performance_on_dataset(pipeline, val_docs, val_targets, val_set_name))

    with open(OUT_PATH + 'multi_language/' + 'performance_report.json', 'w') as fd:
        json.dump(report, fd, indent=2)

    store_model(pipeline, 'multi_language')
    return report, pipeline


def train_language(lang):
    seed = 42
    df_train = get_trainingset(lang)
    val_sets = get_all_validation_sets()
    train_size = 200000

    df_sel = df_train[df_train['target'] == 1].sample(int(train_size / 2), random_state=seed, replace=True)
    df_sel = df_sel.append(df_train[df_train['target'] == 0].sample(int(train_size / 2), random_state=seed, replace=True))

    report, pipeline = run_ml_artifact_training(df_sel, LinearSVC(random_state=42))

    report.update({'seed': seed})
    report.update({'train_frac': train_size})

    for val_set_name, val_set_df in val_sets.items():
        val_docs = val_set_df.copy().pop('doc').values
        val_targets = val_set_df.copy().pop('target').values
        report.update(validation_performance_on_dataset(pipeline, val_docs, val_targets, val_set_name))

    with open(OUT_PATH + lang + '/' + 'performance_report.json', 'w') as fd:
        json.dump(report, fd, indent=2)

    investigate_miscalssifications(pipeline, val_sets[lang + '_researcher_1'], lang + '_researcher_1', lang)

    store_model(pipeline, lang)
    return report, pipeline


def investigate_miscalssifications(pipeline, val_set_df, val_set_name, lang):
    data = val_set_df.copy().pop('doc').values
    target = val_set_df.copy().pop('target').values
    name = val_set_name

    y_predicted = pipeline.predict(data)

    wrongly_identified_as_artifact = []
    wrongly_identified_as_text = []
    for index in range(0, len(data)):
        if target[index] == y_predicted[index]:
            pass
        elif target[index] == TARGET_NAMES['artifact'] and y_predicted[index] == TARGET_NAMES['text']:
            wrongly_identified_as_text.append(data[index])
        else:
            wrongly_identified_as_artifact.append(data[index])

    with open(OUT_PATH + lang + '/' + name + '_wrongly_identified_as_artifact.txt', 'w') as fd:
        fd.write('\n\n'.join(wrongly_identified_as_artifact))
    with open(OUT_PATH + lang + '/' + name + '_wrongly_identified_as_text.txt', 'w') as fd:
        fd.write('\n\n'.join(wrongly_identified_as_text))


def store_model(pipeline, name):
    joblib.dump(pipeline, OUT_PATH + name + '/' + 'artifact_detection.joblib')


if __name__ == "__main__":
    main()
