import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.svm import LinearSVC
from sklearn.utils import resample

from artifact_detection_model.dataset_creation import get_nlon_dataset
from artifact_detection_model.model_training import run_ml_artifact_training
from artifact_detection_model.utils.Logger import Logger
from artifact_detection_model.utils.paths import PROJECT_ROOT, NLON_PATH, NLON_DATASETS

log = Logger()

OUT_PATH = PROJECT_ROOT + "evaluation/out/nlon_bootstrap/"

balance = True
reviewer = 'Fabio'


def get_dataset(file_name):
    df = pandas.read_csv(NLON_PATH + file_name)
    df = df[~df[reviewer].isnull()]
    df['target'] = df[reviewer].replace(2, 0)
    df['doc'] = df['Text']
    return df[['doc', 'target']]


def main():
    df = pandas.DataFrame()
    for name, csv_file in NLON_DATASETS:
        log.s(csv_file)
        doc, targ = get_nlon_dataset(NLON_PATH + csv_file, 'Fabio', balance=True)
        df = df.append(pandas.DataFrame({'doc': doc, 'target':targ}))

    docs = df.copy().pop('doc').values
    target = df.copy().pop('target').values

    n_iterations = 10
    # n_iterations = 100
    n_size = int(len(docs)*0.8)

    df = pandas.DataFrame()
    for i in range(n_iterations):
        # prepare train and test sets
        docs_indices = list(range(0, len(docs)))
        train_idx, t_ = resample(docs_indices, target, n_samples=n_size, stratify=target)
        train_x = docs[train_idx]
        train_y = target[train_idx]

        test_idx = [x for x in docs_indices if x not in list(train_idx)]
        test_x = docs[test_idx]
        test_y = target[test_idx]

        df_train = pandas.DataFrame({'doc': train_x, 'target': train_y})
        df_test = pandas.DataFrame({'doc': test_x, 'target': test_y})

        # fit model
        report, _ = run_ml_artifact_training(df_train, df_test, LinearSVC(random_state=42))
        df = df.append(pandas.DataFrame([report]))

    df.to_csv(OUT_PATH + 'reports.csv')
    evaluate_bootstrap(df, 'macro_f1')
    evaluate_bootstrap(df, 'roc-auc')


def evaluate_bootstrap(df, metric):
    df[metric].plot(kind='hist')
    plt.savefig(OUT_PATH + metric + '.png')
    plt.close()

    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(df[metric], p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(df[metric], p))
    mean = df[metric].mean()
    pandas.DataFrame([{'alpha': alpha*100, 'lower': lower*100, 'upper': upper*100, 'mean': mean}]).to_csv(OUT_PATH + metric + '.csv')
    print(metric + ': %.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))


if __name__ == "__main__":
    main()
