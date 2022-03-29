import json
import numpy as np
import pandas
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from artifact_detection_model.utils.Logger import Logger
from datasets.constants import LANGUAGES
from evaluation.stats_utils import evaluate_bootstrap, t_test_x_greater_y, get_box
from file_anchor import root_dir


log = Logger()

CROSS_LANGUAGE_EVALUATION = root_dir() + 'evaluation/out/cross_language/'

OUT_PATH = root_dir() + 'evaluation/out/single_language/'

language_labels = {
    'cpp': 'C++',
    'java': 'Java',
    'javascript': 'JavaScript',
    'php': 'PHP',
    'python': 'Python',
}


def main():
    performance_table()
    roc_auc_boxplots_both_VS_sets()
    plot_bootstrap_boxdiagram()
    p_test_language_scores('1')


def roc_auc_boxplots_both_VS_sets():
    results_df = pandas.DataFrame()
    rs1_boxes = []
    rs2_boxes = []
    for lang in LANGUAGES:
        lang_df = pandas.read_csv(CROSS_LANGUAGE_EVALUATION + lang + '_artifact_detection_cross_language_resample_summary.csv')

        lang_res = evaluate_bootstrap(lang_df['roc-auc_' + lang + '_researcher_1'], lang + '_VS1')
        results_df = results_df.append(pandas.DataFrame([lang_res]))
        rs1_boxes.append(get_box(lang_res))

        lang_res = evaluate_bootstrap(lang_df['roc-auc_' + lang + '_researcher_2'], lang + '_VS2')
        results_df = results_df.append(pandas.DataFrame([lang_res]))
        rs2_boxes.append(get_box(lang_res))


    results_df.to_csv(OUT_PATH + 'single_language_roc_auc_bootstrap_both_validation_sets.csv')
    fig, ax1 = plt.subplots(figsize=(8, 4))

    space = 0.2 # boxprops=dict(facecolor='tab:blue'),
    boxplot1 = ax1.bxp(rs1_boxes, showfliers=False, widths=0.4, patch_artist=True, boxprops=dict(facecolor='lightcyan'), medianprops=dict(color="black", linewidth=1.5), positions=np.arange(5)-space,)
    ax2 = ax1.twinx()
    boxplot2 = ax2.bxp(rs2_boxes, showfliers=False, widths=0.4, patch_artist=True, boxprops=dict(facecolor='lightgreen'), medianprops=dict(color="black", linewidth=1.5), positions=np.arange(5)+space,)

    ax1_lim = ax1.get_ylim()
    ax2_lim = ax2.get_ylim()

    ax1.set_ylim(0.90, 0.97)
    ax2.set_ylim(0.90, 0.97)
    ax2.set_yticks([])

    ax1.set_xticks(np.arange(5))
    ax1.set_xticklabels([f'{label}' for label in language_labels.values()])

    ax1.set_ylabel('ROC-AUC')
    ax1.set_title('')
    plt.sca(ax1)
    plt.legend(handles=[mpatches.Patch(color='lightcyan', label='Validation set 1'),
                        mpatches.Patch(color='lightgreen', label='Validation set 2')],
               loc='lower left')

    plt.tight_layout()
    plt.savefig(OUT_PATH + 'single_language_roc_auc_boxplot_both_validation_sets.pdf')


def plot_bootstrap_boxdiagram():
    boxes = []
    for lang in LANGUAGES:
        lang_df = pandas.read_csv(CROSS_LANGUAGE_EVALUATION + lang + '_artifact_detection_cross_language_resample_summary.csv')

        lang_res = evaluate_bootstrap(lang_df['roc-auc_' + lang + '_researcher_1'], language_labels[lang])
        boxes.append(get_box(lang_res))

    fig, ax = plt.subplots(figsize=(8, 4))

    boxplot = ax.bxp(boxes, showfliers=False, widths=None) #  patch_artist=True, medianprops=dict(color="black", linewidth=1.5),

    ax.set_ylabel('ROC-AUC')
    ax.set_title('')
    plt.sca(ax)
    plt.tight_layout()
    plt.savefig(OUT_PATH + 'single_language_roc_auc_boxplot.pdf')


def performance_table():
    df = pandas.DataFrame()
    for lang in LANGUAGES:
        lang_df = pandas.read_csv(CROSS_LANGUAGE_EVALUATION + lang + '_artifact_detection_cross_language_resample_summary.csv')
        for researcher in ['1', '2']:
            timeit_col = 'timeit_runtime_' + lang + '_researcher_' + researcher
            valset_size_col = 'man_validation_samples_' + lang + '_researcher_' + researcher
            lang_df[timeit_col] = lang_df[timeit_col]*5000/lang_df[valset_size_col]

            lang_df['art_recall_' + lang + '_researcher_' + researcher] = \
                lang_df['classification_report_' + lang + '_researcher_' + researcher].apply(lambda x: json.loads(x)['artifact']['recall'])
            lang_df['art_precision_' + lang + '_researcher_' + researcher] = \
                lang_df['classification_report_' + lang + '_researcher_' + researcher].apply(lambda x: json.loads(x)['artifact']['precision'])

        columns = {'roc-auc_' + lang + '_researcher_' + x: 'ROC-AUC Reseacher ' + x for x in ['1', '2']}
        columns.update({'art_precision_' + lang + '_researcher_' + x: 'Artifact precision Researcher ' + x for x in ['1', '2']})
        columns.update({'art_recall_' + lang + '_researcher_' + x: 'Artifact recall Researcher ' + x for x in ['1', '2']})
        columns.update({'timeit_runtime_' + lang + '_researcher_1': 'Prediction time per 5000 lines (s)'})
        columns.update({'perf_train_runtime': 'Training time', 'model_size': 'Model size (MiB)'})

        perf_mean_df = lang_df.rename(columns=columns)[list(columns.values())].mean().T
        perf_mean_df['Language'] = language_labels[lang]

        df = df.append(perf_mean_df, ignore_index=True)

    df = df[['Language',
             'ROC-AUC Reseacher 1',
             'ROC-AUC Reseacher 2',
             'Artifact recall Researcher 1',
             'Artifact recall Researcher 2',
             'Artifact precision Researcher 1',
             'Artifact precision Researcher 2',
             'Model size (MiB)',
             'Training time',
             'Prediction time per 5000 lines (s)']]
    df = df.set_index('Language')
    df.to_csv(OUT_PATH + '200k_single_lang_performance_table.csv')
    df.T.to_latex(OUT_PATH + '200k_single_lang_performance_table.tex', float_format="%.2f")


def p_test_language_scores(validation_set_no):
    rep_df = pandas.DataFrame()

    for lang1, lang2 in [('cpp', 'java'), ('java', 'javascript'), ('javascript', 'php'), ('php', 'python')]:
        df1 = pandas.read_csv(CROSS_LANGUAGE_EVALUATION + lang1 + '_artifact_detection_cross_language_resample_summary.csv')
        df2 = pandas.read_csv(CROSS_LANGUAGE_EVALUATION + lang2 + '_artifact_detection_cross_language_resample_summary.csv')

        rep = t_test_x_greater_y(df1['roc-auc_' + lang1 + '_researcher_' + validation_set_no],
                                 df2['roc-auc_' + lang2 + '_researcher_' + validation_set_no],
                                 lang1, lang2)  # one sided, x greater y
        rep['model'] = lang1 + ' vs ' + lang2
        rep_df = rep_df.append(rep)

    rep_df.to_csv(OUT_PATH + 'lang1_better_lang_2_VS' + validation_set_no + '.csv')


if __name__ == "__main__":
    main()
