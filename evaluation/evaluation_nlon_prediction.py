import matplotlib.pyplot as plt
import pandas
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score
from sklearn.metrics import f1_score

from artifact_detection_model.constants import TARGET_NAMES
from artifact_detection_model.utils.Logger import Logger
from artifact_detection_model.utils.paths import PROJECT_ROOT
from evaluation.krippendorff import krippendorff
from evaluation.utils import plot_numpy_confusion_matrix

nlon_prediction_path = PROJECT_ROOT + 'evaluation/out/nlon_predictions/'

log = Logger()


def main():
    pretrained_nlon_on_validation_set()
    nlon_fully_trained_test_set()
    nlon_fully_trained_validation_set()


def nlon_fully_trained_test_set():
    df = pandas.read_csv(nlon_prediction_path + 'full_test_set_nlon_predicted.csv')
    validation(df.pop('target').values, df.pop('nlonPrediction').values, 'target', 'NLoN')


def nlon_fully_trained_validation_set():
    df = pandas.read_csv(nlon_prediction_path + 'full_validation_set_nlon_predicted.csv')
    validation(df.pop('target').values, df.pop('nlonPrediction').values, 'Reseracher2', 'NLoN')


def pretrained_nlon_on_validation_set():
    df = pandas.read_csv(nlon_prediction_path + 'pretrained_nlon_predict_validation_set.csv')
    df.loc[df['nlonPrediction'] == 'Not', 'nlonPrediction'] = 0
    df.loc[df['nlonPrediction'] == 'NL', 'nlonPrediction'] = 1
    df['nlonPrediction'] = df['nlonPrediction'].astype(int)
    validation(df.pop('target').values, df.pop('nlonPrediction').values, 'Reseracher2', 'NLoNpretrained')


def validation(r1_target, r2_target, r1name, r2name):
    outfile = r1name + '_vs_' + r2name
    cm = confusion_matrix(r1_target, r2_target)
    disp = plot_numpy_confusion_matrix(cm, list(TARGET_NAMES.keys()))
    disp.ax_.set_ylabel(r1name)
    disp.ax_.set_xlabel(r2name)
    plt.gcf().subplots_adjust(left=0.2)
    plt.tight_layout()
    plt.savefig(nlon_prediction_path + 'confusion_matrix_' + outfile + '.png')

    ir_metrics = [{'cohens_kappa': cohen_kappa_score(r1_target, r2_target),
                   'weighted_f1': f1_score(r1_target, r2_target, average='weighted'),
                   'macro_f1': f1_score(r1_target, r2_target, average='macro'),
                   'accuracy': accuracy_score(r1_target, r2_target),
                   'krippendorff_alpha': krippendorff.alpha([r1_target, r2_target]),
                   'roc_auc_w_mica_as_truth': roc_auc_score(r1_target, r2_target)}]
    pandas.DataFrame(ir_metrics).to_csv(nlon_prediction_path + outfile + '.csv')

    print('cohen ' + str(cohen_kappa_score(r1_target, r2_target)))
    print('f1 ' + r1name + ' base ' + str(f1_score(r1_target, r2_target, average='weighted')))
    print('f1 ' + r2name + ' base ' + str(f1_score(r2_target, r1_target, average='weighted')))
    print('accuracy ' + str(accuracy_score(r1_target, r2_target)))
    print('krippendorff alpha ' + str(krippendorff.alpha([r1_target, r2_target])))
    print('roc auc ' + r1name + ' base ' + str(roc_auc_score(r1_target, r2_target)))
    print('roc auc ' + r2name + ' base ' + str(roc_auc_score(r2_target, r1_target)))


if __name__ == '__main__':
    main()
