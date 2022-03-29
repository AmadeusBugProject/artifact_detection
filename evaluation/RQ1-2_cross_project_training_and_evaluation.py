import random
import pandas
from sklearn.svm import LinearSVC

from artifact_detection_model.model_training import run_ml_artifact_training
from artifact_detection_model.utils.Logger import Logger
from datasets.constants import LANGUAGES
from datasets.dataset_utils import get_trainingset, get_all_validation_sets
from evaluation.utils import validation_performance_on_dataset
from file_anchor import root_dir

log = Logger()

OUT_PATH = root_dir() + "evaluation/out/cross_language/"


def main():
    val_sets = get_all_validation_sets()

    for lang in LANGUAGES:
        df = evaluate_language_model(lang, val_sets)


def evaluate_language_model(lang, val_sets, train_size=200000):
    df_train = get_trainingset(lang, balance=False)

    n_iterations = 100
    df = pandas.DataFrame()

    for index in range(n_iterations):
        seed = random.randint(100, 1000)

        df_sel = df_train[df_train['target'] == 1].sample(int(train_size/2), random_state=seed, replace=True)
        df_sel = df_sel.append(df_train[df_train['target'] == 0].sample(int(train_size/2), random_state=seed, replace=True))

        report, pipeline = run_ml_artifact_training(df_sel, LinearSVC(random_state=42))
        report.update({'seed': seed})
        report.update({'train_frac': train_size})
        report.update({'index': index})

        for val_set_name, val_set_df in val_sets.items():
            val_docs = val_set_df.copy().pop('doc').values
            val_targets = val_set_df.copy().pop('target').values
            report.update(validation_performance_on_dataset(pipeline, val_docs, val_targets, val_set_name))
        print(report)

        df = df.append(pandas.DataFrame([report]))

    df.to_csv(OUT_PATH + lang + '_artifact_detection_cross_language_resample_summary.csv')
    return df


if __name__ == "__main__":
    main()
