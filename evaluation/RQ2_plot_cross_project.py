import numpy as np
import pandas
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from artifact_detection_model.utils.Logger import Logger
from datasets.constants import LANGUAGES
from evaluation.stats_utils import t_test_x_greater_y
from file_anchor import root_dir

import seaborn as sns

log = Logger()

OUT_PATH = root_dir() + 'evaluation/out/cross_language/'

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
        p_test_model_trained_performs_better_on_its_own_language_than_other_languages(validation_set_no)


def bare_stats():
    for lang in LANGUAGES:
        df = pandas.read_csv(OUT_PATH + lang + '_artifact_detection_cross_language_resample_summary.csv')
        columns = ['roc-auc_' + lang + '_researcher_' + x for x in ['1', '2']]
        df[columns].describe().to_csv(OUT_PATH + lang + '_performance.csv')


def p_test_model_trained_performs_better_on_its_own_language_than_other_languages(validation_set_no):
    rep_df = pandas.DataFrame()
    for lang in LANGUAGES:
        df = pandas.read_csv(OUT_PATH + lang + '_artifact_detection_cross_language_resample_summary.csv')

        for l in LANGUAGES:
            if lang == l:
                continue
            rep = t_test_x_greater_y(df['roc-auc_' + lang + '_researcher_' + validation_set_no],
                                     df['roc-auc_' + l + '_researcher_' + validation_set_no],
                                     lang, l)  # one sided, x greater y
            rep['model'] = lang
            rep_df = rep_df.append(rep)

    rep_df.to_csv(OUT_PATH + 'cross_project_model_better_on_its_own_language_than_other_languages_VS' + validation_set_no + '.csv')


def cross_project_roc_auc_matrix(validation_set_no):
    columns = ['roc-auc_' + x + '_researcher_' + validation_set_no for x in LANGUAGES]
    cm = []
    for lang in LANGUAGES:
        df = pandas.read_csv(OUT_PATH + lang + '_artifact_detection_cross_language_resample_summary.csv')
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
                yticklabels=[language_labels[x] for x in LANGUAGES])
    plt.yticks(rotation=0)
    ax.set(ylabel="Model language", xlabel='Validation set ' + validation_set_no + ' language', title='ROC-AUC')

    plt.tight_layout()
    plt.savefig(OUT_PATH + 'cross_project_roc_auc_matrix_VS' + validation_set_no + '.pdf')


def plot_numpy_confusion_matrix(cm, target_names):
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=target_names)
    disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal', values_format=None)
    return disp


if __name__ == "__main__":
    main()
