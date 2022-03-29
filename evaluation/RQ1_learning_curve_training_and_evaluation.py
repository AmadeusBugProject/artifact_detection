import random
import traceback

import pandas
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC

from artifact_detection_model.model_training import run_ml_artifact_training
from artifact_detection_model.utils.Logger import Logger
from datasets.constants import LANGUAGES
from datasets.dataset_utils import get_trainingset, get_validation_sets_for_language
from evaluation.utils import validation_performance_on_dataset
from file_anchor import root_dir

log = Logger()

OUT_PATH = root_dir() + 'evaluation/out/learning_curve/'


def main():
    for lang in LANGUAGES:
        df = get_learning_curve_data(lang)
        df = pandas.read_csv(OUT_PATH + lang + '_artifact_detection_summary.csv')
        plot_learning_curve(df, lang)
        scoring_report(df, lang)


def scoring_report(df, lang):
    df = df[df['train_frac'] == 25000]
    df.mean().to_csv(OUT_PATH + lang + '_means.csv')


def plot_learning_curve(df, language):
    gb = df.groupby(by='train_samples')

    # validation set
    fig, axes = plt.subplots(1, 1, figsize=(5, 4))
    plot_mean_and_fill_std(axes, gb, 'macro_f1_' + language + '_researcher_1', 'g', 'Validation set 1')
    plot_mean_and_fill_std(axes, gb, 'macro_f1_' + language + '_researcher_2', 'b', 'Validation set 2')
    axes.set_ylabel('F1')
    axes.set_xlabel('Training set size')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH + language + '_macro_f1_validation_set_learning_curve.png')
    plt.close()

    fig, axes = plt.subplots(1, 1, figsize=(5, 4))
    plot_mean_and_fill_std(axes, gb, 'roc-auc_' + language + '_researcher_1', 'g', 'Validation set 1')
    plot_mean_and_fill_std(axes, gb, 'roc-auc_' + language + '_researcher_2', 'b', 'Validation set 2')
    axes.set_ylabel('ROC-AUC')
    axes.set_xlabel('Training set size')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(OUT_PATH + language + '_roc-auc_validation_set_learning_curve.png')
    plt.close()

    # model size
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    plot_mean_and_fill_std(axes, gb, 'model_size', 'r', 'Model size (MiB)')
    axes.set_ylabel('MiB')
    axes.set_xlabel('Training set size')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH + language + '_model_size_learning_curve.png')
    plt.close()

    # runtime performance
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    plot_mean_and_fill_std(axes, gb, 'perf_train_runtime', 'r', 'Training time')
    axes.set_ylabel('Seconds')
    axes.set_xlabel('Training set size')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH + language + '_perf_train_runtime_learning_curve.png')
    plt.close()

    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    plot_mean_and_fill_std(axes, gb, 'perf_predict_runtime_' + language + '_researcher_1', 'r', 'Validation set 1 classification time (perf_counter)')
    axes.set_ylabel('Seconds')
    axes.set_xlabel('Training set size')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH + language + '_perf_predict_runtime_learning_curve.png')
    plt.close()

    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    plot_mean_and_fill_std(axes, gb, 'timeit_runtime_' + language + '_researcher_1', 'r', 'Validation set 1 classification time (timeit)')
    axes.set_ylabel('Seconds')
    axes.set_xlabel('Training set size')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH + language + '_perf_timeit_predict_runtime_learning_curve.png')
    plt.close()


def plot_mean_and_fill_std(axes, gb, metric, color, label):
    axes.fill_between(gb.mean().index, gb.mean()[metric] - gb.std()[metric],
                         gb.mean()[metric] + gb.std()[metric], alpha=0.1,
                         color=color)
    axes.plot(gb.mean().index, gb.mean()[metric], 'o-', color=color, label=label)


def get_learning_curve_data(lang):
    df_train = get_trainingset(lang, balance=False)
    val_sets = get_validation_sets_for_language(lang)

    df = pandas.DataFrame()

    try:
        for train_frac in [6250, 12500, 25000, 50000, 100000, 200000, 400000, 800000, 1600000, 3200000]:
            if train_frac > len(df_train):
                act_train_frac = len(df_train)
            else:
                act_train_frac = train_frac
            for index in range(0, 10):
                seed = random.randint(100, 1000)

                df_sel = df_train[df_train['target'] == 1].sample(int(act_train_frac / 2), random_state=seed, replace=True)
                df_sel = df_sel.append( df_train[df_train['target'] == 0].sample(int(act_train_frac / 2), random_state=seed, replace=True))

                report, pipeline = run_ml_artifact_training(df_sel, LinearSVC(random_state=42))
                report.update({'seed': seed})
                report.update({'train_frac': act_train_frac})
                report.update({'index': index})

                for val_set_name, val_set_df in val_sets.items():
                    val_docs = val_set_df.copy().pop('doc').values
                    val_targets = val_set_df.copy().pop('target').values
                    report.update(validation_performance_on_dataset(pipeline, val_docs, val_targets, val_set_name))
                print(report)

                df = df.append(pandas.DataFrame([report]))
            if train_frac > len(df_train):
                break
    except Exception as e:
        log.e(str(e))
        log.e(str(traceback.format_tb(e.__traceback__)))

    df.to_csv(OUT_PATH + lang + '_artifact_detection_summary.csv')
    return df


if __name__ == "__main__":
    main()
