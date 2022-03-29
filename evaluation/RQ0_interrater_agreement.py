import pandas
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, cohen_kappa_score, f1_score, \
    accuracy_score

from artifact_detection_model.utils.Logger import Logger

from datasets.constants import LANGUAGES
from datasets.dataset_utils import get_all_validation_sets
from evaluation.krippendorff import krippendorff
from evaluation.utils import plot_numpy_confusion_matrix
from file_anchor import root_dir

log = Logger()

out_path = root_dir() + 'evaluation/out/interrater_agreement/'
target_names = {'artifact': 0,
                'text': 1}

language_labels = {
    'cpp': 'C++',
    'java': 'Java',
    'javascript': 'JavaScript',
    'php': 'PHP',
    'python': 'Python',
}


def combine_data_sets(reviewer1_df, reviewer2_df):
    reviewer1_df.rename(columns={'doc': 'doc1', 'target': 'target1'}, inplace=True)
    reviewer2_df.rename(columns={'doc': 'doc2', 'target': 'target2'}, inplace=True)
    combined = pandas.concat([reviewer1_df, reviewer2_df], axis=1)
    combined = combined[~combined['target1'].isnull()]
    combined = combined[~combined['target2'].isnull()]
    combined = combined[~combined['doc1'].isnull()]
    combined = combined[~combined['doc2'].isnull()]
    combined['target1'] = combined['target1'].astype('float')
    combined['target2'] = combined['target2'].astype('float')
    return combined


def main():
    val_sets = get_all_validation_sets()
    latex_table_interrrater_agreement(val_sets)
    for lang in LANGUAGES:
        r_1_df = val_sets[lang + '_researcher_1']
        r_2_df = val_sets[lang + '_researcher_2']
        interrater_agreement_per_language(lang, r_1_df, r_2_df)


def latex_table_interrrater_agreement(val_sets):
    aa = {'Researcher 1': 'artifact', 'Researcher 2': 'artifact'}
    an = {'Researcher 1': 'artifact', 'Researcher 2': 'natural language'}
    na = {'Researcher 1': 'natural language', 'Researcher 2': 'artifact'}
    nn = {'Researcher 1': 'natural language', 'Researcher 2': 'natural language'}

    for lang in LANGUAGES:
        r_1_df = val_sets[lang + '_researcher_1']
        r_2_df = val_sets[lang + '_researcher_2']
        df = combine_data_sets(r_1_df, r_2_df)

        aa.update({'Researcher 1': 'artifact', 'Researcher 2': 'artifact', language_labels[lang]: len(df[(df['target1'] == 0) & (df['target2'] == 0)])/len(df)*100})
        an.update({'Researcher 1': 'artifact', 'Researcher 2': 'natural language', language_labels[lang]: len(df[(df['target1'] == 0) & (df['target2'] == 1)])/len(df)*100})
        na.update({'Researcher 1': 'natural language', 'Researcher 2': 'artifact', language_labels[lang]: len(df[(df['target1'] == 1) & (df['target2'] == 0)])/len(df)*100})
        nn.update({'Researcher 1': 'natural language', 'Researcher 2': 'natural language', language_labels[lang]: len(df[(df['target1'] == 1) & (df['target2'] == 1)])/len(df)*100})

    df = pandas.DataFrame([aa, an, na, nn])
    df.to_csv(out_path + 'interrater_stats.csv')
    df.to_latex(out_path + 'interrater_stats.tex', float_format="%.2f")


def interrater_agreement_per_language(language, r_1_df, r_2df):

    all_df = combine_data_sets(r_1_df, r_2df)
    all_df[all_df['target1'] != all_df['target2']].to_csv(out_path + language + '_reviewer1_vs_reviewer2_mismatched.csv')

    r1_target = all_df['target1'].to_list()
    r2_target = all_df['target2'].to_list()

    cm = confusion_matrix(r1_target, r2_target)
    disp = plot_numpy_confusion_matrix(cm, list(target_names.keys()))
    disp.ax_.set_ylabel('Researcher 1')
    disp.ax_.set_xlabel('Researcher 2')
    plt.gcf().subplots_adjust(left=0.2)
    plt.tight_layout()
    plt.savefig(out_path + language + '_confusion_matrix_reviewer1_vs_reviewer2.png')

    ir_metrics = [{'cohens_kappa': cohen_kappa_score(r1_target, r2_target),
                   'weighted_f1': f1_score(r1_target, r2_target, average='weighted'),
                   'macro_f1': f1_score(r1_target, r2_target, average='macro'),
                   'accuracy': accuracy_score(r1_target, r2_target),
                   'krippendorff_alpha': krippendorff.alpha([r1_target, r2_target]),
                   'roc_auc': roc_auc_score(r1_target, r2_target)}]
    pandas.DataFrame(ir_metrics).to_csv(out_path + language + '_reviewer1_vs_reviewer2_manual_agreement.csv')


if __name__ == "__main__":
    main()
