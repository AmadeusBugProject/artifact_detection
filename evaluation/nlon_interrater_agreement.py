import pandas
import matplotlib.pyplot as plt
import pandas
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score
from sklearn.metrics import f1_score

from artifact_detection_model.constants import TARGET_NAMES
from artifact_detection_model.utils.Logger import Logger
from artifact_detection_model.utils.paths import PROJECT_ROOT, NLON_DATASETS, NLON_PATH
from evaluation.krippendorff import krippendorff
from evaluation.utils import plot_numpy_confusion_matrix

OUT_PATH = PROJECT_ROOT + 'evaluation/out/nlon_interrater_agreement/'

log = Logger()


def write_for_exernal_validataion(df):
    external_vali = df.sample(n=2000, random_state=42).copy()
    external_vali['Reviewer1'] = ''
    external_vali[['Reviewer1', 'Text', 'ID', 'src', 'Fabio', 'Mika', 'Disagreement']].to_csv(OUT_PATH + 'random_for_external.csv')


def main():
    # write_for_exernal_validataion()
    nlon_internal_validation()
    nlon_external_validation()


def nlon_external_validation():
    all_df = pandas.read_csv(OUT_PATH + 'random_selection_reviewer_1.csv')

    r1_target = all_df['Mika'].to_list()
    r2_target = all_df['Reviewer1'].to_list()
    validation(r1_target, r2_target, 'Mika', 'Reviewer1')


def nlon_internal_validation():
    all_df = pandas.DataFrame()
    for name, csv_file in NLON_DATASETS:
        log.s(csv_file)
        df = pandas.read_csv(NLON_PATH + csv_file)
        df = df[~df['Mika'].isnull()]
        df = df[~df['Fabio'].isnull()]

        df['Mika'] = df['Mika'].replace(2, 0)
        df['Fabio'] = df['Fabio'].replace(2, 0)
        df['src'] = csv_file
        all_df = all_df.append(df)

    r1_target = all_df['Mika'].to_list()
    r2_target = all_df['Fabio'].to_list()
    validation(r1_target, r2_target, 'Mika', 'Fabio')


def validation(r1_target, r2_target, r1name, r2name):
    outfile = r1name + '_vs_' + r2name
    cm = confusion_matrix(r1_target, r2_target)
    disp = plot_numpy_confusion_matrix(cm, list(TARGET_NAMES.keys()))
    disp.ax_.set_ylabel(r1name)
    disp.ax_.set_xlabel(r2name)
    plt.gcf().subplots_adjust(left=0.2)
    plt.tight_layout()
    plt.savefig(OUT_PATH + 'confusion_matrix_' + outfile + '.png')

    ir_metrics = [{'cohens_kappa': cohen_kappa_score(r1_target, r2_target),
                   'weighted_f1': f1_score(r1_target, r2_target, average='weighted'),
                   'macro_f1': f1_score(r1_target, r2_target, average='macro'),
                   'accuracy': accuracy_score(r1_target, r2_target),
                   'krippendorff_alpha': krippendorff.alpha([r1_target, r2_target]),
                   'roc_auc_w_mica_as_truth': roc_auc_score(r1_target, r2_target)}]
    pandas.DataFrame(ir_metrics).to_csv(OUT_PATH + outfile + '.csv')

    print('cohen ' + str(cohen_kappa_score(r1_target, r2_target)))
    print('f1 ' + r1name + ' base ' + str(f1_score(r1_target, r2_target, average='weighted')))
    print('f1 ' + r2name + ' base ' + str(f1_score(r2_target, r1_target, average='weighted')))
    print('accuracy ' + str(accuracy_score(r1_target, r2_target)))
    print('krippendorff alpha ' + str(krippendorff.alpha([r1_target, r2_target])))
    print('roc auc ' + r1name + ' base ' + str(roc_auc_score(r1_target, r2_target)))
    print('roc auc ' + r2name + ' base ' + str(roc_auc_score(r2_target, r1_target)))


if __name__ == '__main__':
    main()
