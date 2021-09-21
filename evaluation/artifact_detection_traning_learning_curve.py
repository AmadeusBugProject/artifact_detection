import random

import pandas
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC

from artifact_detection_model.dataset_creation import get_manual_validation_data_set, get_training_and_test_set
from artifact_detection_model.model_training import run_ml_artifact_training
from artifact_detection_model.utils.Logger import Logger
from artifact_detection_model.utils.paths import PROJECT_ROOT, REVIEWER_1_PATH, REVIEWER_2_PATH
from evaluation.utils import validation_performance_on_dataset, nlon_performance

log = Logger()

OUT_PATH = PROJECT_ROOT + 'evaluation/out/learning_curve/'


def main():
    df = get_learning_curve_data()
    # df = pandas.read_csv(OUT_PATH + 'artifact_detection_summary.csv')
    plot_learning_curve(df)
    scoring_report(df)


def scoring_report(df):
    df = df[df['train_frac'] == 0.4]
    df.mean().to_csv(OUT_PATH + 'means.csv')


def plot_learning_curve(df):
    df = df[df['train_frac'] > 0.1]
    # df[['index', 'train_frac', 'train_samples', 'macro_f1', 'macro_f1_reviewer_2', 'roc-auc', 'roc-auc_reviewer_2']]
    gb = df.groupby(by='train_samples')

    # validation set
    fig, axes = plt.subplots(1, 1, figsize=(5, 4))
    plot_mean_and_fill_std(axes, gb, 'macro_f1', 'r', 'Test set')
    plot_mean_and_fill_std(axes, gb, 'macro_f1_reviewer_1', 'g', 'Validation set 1')
    plot_mean_and_fill_std(axes, gb, 'macro_f1_reviewer_2', 'b', 'Validation set 2')
    plt.axvline(x=288038, color='gray', label='40% Training set')
    axes.set_ylabel('F1')
    axes.set_xlabel('Training set size')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH + 'macro_f1_validation_set_learning_curve.png')

    fig, axes = plt.subplots(1, 1, figsize=(5, 4))
    plot_mean_and_fill_std(axes, gb, 'roc-auc', 'r', 'Test set')
    plot_mean_and_fill_std(axes, gb, 'roc-auc_reviewer_1', 'g', 'Validation set 1')
    plot_mean_and_fill_std(axes, gb, 'roc-auc_reviewer_2', 'b', 'Validation set 2')
    plt.axvline(x=288038, color='gray', label='40% Training set')
    axes.set_ylabel('ROC-AUC')
    axes.set_xlabel('Training set size')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(OUT_PATH + 'roc-auc_validation_set_learning_curve.png')

    # nlon
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    plot_mean_and_fill_std(axes, gb, 'roc-auc', 'r', 'Test set')
    plot_mean_and_fill_std(axes, gb, 'roc-auc_nlon_all_Fabio', 'b', 'NLoN Fabio')
    plot_mean_and_fill_std(axes, gb, 'roc-auc_nlon_all_Mika', 'g', 'NLoN Mika')
    axes.set_ylabel('ROC-AUC')
    axes.set_xlabel('Training set size')
    plt.axvline(x=288038, color='gray', label='40% Training set')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH + 'roc-auc_nlon_all_set_learning_curve.png')

    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    plot_mean_and_fill_std(axes, gb, 'macro_f1', 'r', 'Test set')
    plot_mean_and_fill_std(axes, gb, 'macro_f1_nlon_all_Fabio', 'b', 'NLoN Fabio')
    plot_mean_and_fill_std(axes, gb, 'macro_f1_nlon_all_Mika', 'g', 'NLoN Mika')
    axes.set_ylabel('F1')
    axes.set_xlabel('Training set size')
    plt.axvline(x=288038, color='gray', label='40% Training set')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH + 'macro_f1_nlon_all_set_learning_curve.png')

    # runtime performance
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    plot_mean_and_fill_std(axes, gb, 'perf_train_runtime', 'r', 'Training time')
    axes.set_ylabel('Seconds')
    axes.set_xlabel('Training set size')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH + 'perf_train_runtime_learning_curve.png')

    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    plot_mean_and_fill_std(axes, gb, 'perf_runtime', 'r', 'Testset classification time')
    axes.set_ylabel('Seconds')
    axes.set_xlabel('Training set size')
    plt.axvline(x=288038, color='gray', label='40% Training set')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH + 'perf_test_set_runtime_learning_curve.png')

    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    plot_mean_and_fill_std(axes, gb, 'timeit_runtime', 'r', 'Testset classification time')
    axes.set_ylabel('Seconds')
    axes.set_xlabel('Training set size')
    plt.axvline(x=288038, color='gray', label='40% Training set')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH + 'perf_timeit_test_set_runtime_learning_curve.png')


def plot_mean_and_fill_std(axes, gb, metric, color, label):
    axes.fill_between(gb.mean().index, gb.mean()[metric] - gb.std()[metric],
                         gb.mean()[metric] + gb.std()[metric], alpha=0.1,
                         color=color)
    axes.plot(gb.mean().index, gb.mean()[metric], 'o-', color=color, label=label)


def get_learning_curve_data():
    r1data, r1target = get_manual_validation_data_set(REVIEWER_1_PATH)
    r2data, r2target = get_manual_validation_data_set(REVIEWER_2_PATH)

    df = pandas.DataFrame()
    df_train, df_test = get_training_and_test_set()

    for train_frac in [0.2, 0.4, 0.6, 0.8, 1]:
        for index in range(0, 10):
            seed = random.randint(100, 1000)
            report, pipeline = run_ml_artifact_training(df_train.copy().sample(frac=train_frac, random_state=seed),
                                                        df_test.copy(),
                                                        LinearSVC(random_state=seed))
            report.update({'seed': seed})
            report.update({'train_frac': train_frac})
            report.update({'index': index})
            report.update(validation_performance_on_dataset(pipeline, r1data, r1target, 'reviewer_1'))
            report.update(validation_performance_on_dataset(pipeline, r2data, r2target, 'reviewer_2'))
            report.update(nlon_performance(pipeline, 'Fabio'))
            report.update(nlon_performance(pipeline, 'Mika'))
            report.update(nlon_performance(pipeline, 'agreement'))
            print(report)

            df = df.append(pandas.DataFrame([report]))

    df.to_csv(OUT_PATH + 'artifact_detection_summary.csv')
    return df


if __name__ == "__main__":
    main()
