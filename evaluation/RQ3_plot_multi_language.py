import numpy as np
import pandas
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from artifact_detection_model.utils.Logger import Logger
from datasets.constants import LANGUAGES
from evaluation.stats_utils import evaluate_bootstrap, t_test_x_greater_y, t_test_x_differnt_y, get_box
from file_anchor import root_dir

import seaborn as sns

log = Logger()

OUT_PATH = root_dir() + 'evaluation/out/multi_language/'
CROSS_LANGUAGE_EVALUATION = root_dir() + 'evaluation/out/cross_language/'

language_labels = {
    'cpp': 'C++',
    'java': 'Java',
    'javascript': 'JavaScript',
    'php': 'PHP',
    'python': 'Python',
}


def main():
    bare_stats()
    for validation_set_no in ['1', '2']:
        cross_project_roc_auc_matrix(validation_set_no)
        p_test_single_lang_model_performs_better_than_multi_lang_model(validation_set_no)
        p_test_single_lang_model_different_than_multi_lang_model(validation_set_no)
        p_test_single_lang_model_different_than_multi_lang_model(validation_set_no)
        roc_auc_boxplots(validation_set_no)
        multi_model_transferability_table(validation_set_no)


def bare_stats():
    multi_df = pandas.read_csv(OUT_PATH + 'artifact_detection_multi_language_model_resample_summary.csv')
    for lang in LANGUAGES:
        columns = ['roc-auc_' + lang + '_researcher_' + x for x in ['1', '2']]
        multi_df[columns].describe().to_csv(OUT_PATH + lang + '_performance.csv')


def p_test_single_lang_model_performs_better_than_multi_lang_model(validation_set_no):
    multi_df = pandas.read_csv(OUT_PATH + 'artifact_detection_multi_language_model_resample_summary.csv')

    rep_df = pandas.DataFrame()
    for lang in LANGUAGES:
        df = pandas.read_csv(CROSS_LANGUAGE_EVALUATION + lang + '_artifact_detection_cross_language_resample_summary.csv')

        rep = t_test_x_greater_y(df['roc-auc_' + lang + '_researcher_' + validation_set_no],
                                 multi_df['roc-auc_' + lang + '_researcher_' + validation_set_no],
                                 lang, 'multilang')  # one sided, x greater y
        rep['model'] = 'multilang'
        rep_df = rep_df.append(rep)

    rep_df.to_csv(OUT_PATH + 'single_lang_model_better_on_its_own_language_than_multilang_model_VS' + validation_set_no + '.csv')


def p_test_single_lang_model_different_than_multi_lang_model(validation_set_no):
    multi_df = pandas.read_csv(OUT_PATH + 'artifact_detection_multi_language_model_resample_summary.csv')

    rep_df = pandas.DataFrame()
    for lang in LANGUAGES:
        df = pandas.read_csv(CROSS_LANGUAGE_EVALUATION + lang + '_artifact_detection_cross_language_resample_summary.csv')

        rep = t_test_x_differnt_y(df['roc-auc_' + lang + '_researcher_' + validation_set_no],
                                 multi_df['roc-auc_' + lang + '_researcher_' + validation_set_no],
                                 lang, 'multilang')  # one sided, x greater y
        rep['model'] = 'multilang'
        rep_df = rep_df.append(rep)

    rep_df.to_csv(OUT_PATH + 'single_lang_model_than_multilang_model_VS' + validation_set_no + '.csv')


def cross_project_roc_auc_matrix(validation_set_no):
    columns = ['roc-auc_' + x + '_researcher_' + validation_set_no for x in LANGUAGES]
    cm = []
    for lang in LANGUAGES:
        df = pandas.read_csv(CROSS_LANGUAGE_EVALUATION + lang + '_artifact_detection_cross_language_resample_summary.csv')
        df = df[columns].mean()
        cm.append(df.to_list())

    df = pandas.read_csv(OUT_PATH + 'artifact_detection_multi_language_model_resample_summary.csv')
    df = df[columns].mean()
    cm.append(df.to_list())

    fig, ax = plt.subplots() #figsize=(3, 3)
    sns.heatmap(cm,
                ax=ax,
                # linewidths=0.01,
                # linecolor='k',
                cmap="viridis",
                annot=True,
                annot_kws={'fontsize':'large'},
                xticklabels=[language_labels[x] for x in LANGUAGES],
                yticklabels=[language_labels[x] for x in LANGUAGES] + ['Multi language'])
    plt.yticks(rotation=0)
    ax.set(ylabel="Model language", xlabel='Validation set ' + validation_set_no + ' language', title='ROC-AUC')

    plt.tight_layout()
    plt.savefig(OUT_PATH + 'multi_language_project_roc_auc_matrix_VS' + validation_set_no + '.pdf')


def roc_auc_boxplots(validation_set_no):
    multi_df = pandas.read_csv(OUT_PATH + 'artifact_detection_multi_language_model_resample_summary.csv')

    results_df = pandas.DataFrame()
    multi_boxes = []
    lang_boxes = []
    for lang in LANGUAGES:
        lang_df = pandas.read_csv(CROSS_LANGUAGE_EVALUATION + lang + '_artifact_detection_cross_language_resample_summary.csv')
        mult_res = evaluate_bootstrap(multi_df['roc-auc_' + lang + '_researcher_' + validation_set_no], 'Multilang' + '_VS' + validation_set_no)
        lang_res = evaluate_bootstrap(lang_df['roc-auc_' + lang + '_researcher_' + validation_set_no], lang + '_VS' + validation_set_no)
        results_df = results_df.append(pandas.DataFrame([mult_res]))
        results_df = results_df.append(pandas.DataFrame([lang_res]))
        multi_boxes.append(get_box(mult_res))
        lang_boxes.append(get_box(lang_res))

    results_df.to_csv(OUT_PATH + 'multi_language_roc_auc_bootstrap_VS' + validation_set_no + '.csv')
    fig, ax1 = plt.subplots(figsize=(8, 4))

    space = 0.2 # boxprops=dict(facecolor='tab:blue'),
    boxplot1 = ax1.bxp(multi_boxes, showfliers=False, widths=0.4, patch_artist=True, boxprops=dict(facecolor='lightcyan'), medianprops=dict(color="black", linewidth=1.5), positions=np.arange(5)-space,)
    ax2 = ax1.twinx()
    boxplot2 = ax2.bxp(lang_boxes, showfliers=False, widths=0.4, patch_artist=True, boxprops=dict(facecolor='lightgreen'), medianprops=dict(color="black", linewidth=1.5), positions=np.arange(5)+space,)

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

    plt.legend(handles=[mpatches.Patch(color='lightcyan', label='Multi language model'),
                        mpatches.Patch(color='lightgreen', label='Language specific model')],
               loc='lower left')

    plt.tight_layout()
    plt.savefig(OUT_PATH + 'multi_language_roc_auc_boxplots_VS' + validation_set_no + '.pdf')


def multi_model_transferability_table(validation_set_no):
    comb_roc_auc = pandas.DataFrame(columns=[language_labels[l] for l in LANGUAGES] + ['Multi language'])

    for lang in LANGUAGES:
        df = pandas.read_csv(CROSS_LANGUAGE_EVALUATION + lang + '_artifact_detection_cross_language_resample_summary.csv')
        roc_auc = []
        for l in LANGUAGES:
            roc_auc.extend(df['roc-auc_' + l + '_researcher_' + validation_set_no].to_list())
        comb_roc_auc[language_labels[lang]] = roc_auc

    df = pandas.read_csv(OUT_PATH + 'artifact_detection_multi_language_model_resample_summary.csv')
    roc_auc = []
    for l in LANGUAGES:
        roc_auc.extend(df['roc-auc_' + l + '_researcher_' + validation_set_no].to_list())
    comb_roc_auc['Multi language'] = roc_auc

    rep_df = pandas.DataFrame()
    for column in comb_roc_auc.columns:
        if column == 'Multi language':
            continue
        rep = t_test_x_greater_y(comb_roc_auc['Multi language'],
                                 comb_roc_auc[column],
                                 'Multi language', column)

        rep_df = rep_df.append(rep)
    rep_df.to_csv(OUT_PATH + 'multilang_model_better_transfer_than_single_lang_model_transfer_VS' + validation_set_no + '.csv')
    rep_df = rep_df[rep_df['test'] == 'wilcoxon']

    comb_roc_auc = comb_roc_auc.mean()
    comb_roc_auc.to_csv(OUT_PATH + 'transferability_mean_over_all_language_performance_VS'+ validation_set_no + '.csv')
    comb_roc_auc.T.to_latex(OUT_PATH + 'transferability_mean_over_all_language_performance_VS' + validation_set_no + '.tex', float_format="%.2f")


if __name__ == "__main__":
    main()
