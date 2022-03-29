import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas


def t_test_x_differnt_y(x, y, x_label, y_label, output_path=None, df_file=None): # two sided
    df = pandas.DataFrame()
    df = df.append(is_normal(x, x_label, output_path))
    df = df.append(is_normal(y, y_label, output_path))

    stat, p = stats.ttest_ind(x, y, equal_var=False, alternative='two-sided')
    h0 = x_label + ' is not different ' + y_label
    df = df.append(pandas.DataFrame({'h0': [h0], 'test': ['t_test_two_sided'], 'stat': [stat], 'p': [p]}))
    if output_path and df_file:
        df.to_csv(output_path + df_file)
    return df


def t_test_x_greater_y(x, y, x_label, y_label, output_path=None, df_file=None): # one sided, x greater y
    df = pandas.DataFrame()
    df = df.append(is_normal(x, x_label, output_path))
    df = df.append(is_normal(y, y_label, output_path))

    stat, p = stats.ttest_ind(x, y, equal_var=False, alternative='greater')
    h0 = x_label + ' is not greater than ' + y_label
    df = df.append(pandas.DataFrame({'h0': [h0], 'test': ['t_test_one_sided'], 'stat': [stat], 'p': [p]}))

    stat, p = stats.wilcoxon(x, y, alternative='greater')
    h0 = x_label + ' is not greater than ' + y_label
    df = df.append(pandas.DataFrame({'h0': [h0], 'test': ['wilcoxon'], 'stat': [stat], 'p': [p]}))
    df['x_label'] = x_label
    df['y_label'] = y_label
    if output_path and df_file:
        df.to_csv(output_path + df_file)
    return df


def is_normal(series, add_label, output_path=None):
    if output_path:
        stats.probplot(series, dist="norm", plot=plt)
        plt.savefig(output_path + add_label + '_normality.png')
        plt.close()
    shapiro_stat, shapiro_p = stats.shapiro(series)
    dagostino_stat, dagostino_p = stats.normaltest(series)

    df = pandas.DataFrame({'test': ['shapiro_stat', 'dagostino_stat'], 'stat': [shapiro_stat, dagostino_stat], 'p': [shapiro_p, dagostino_p]})
    df['h0'] = add_label + ' - that the data was drawn from normal distribution'
    return df


def evaluate_bootstrap(series, label):
    mean = series.mean()
    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(series, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(series, p))

    return {'alpha': alpha * 100,
                           'lower': lower * 100,
                           'upper': upper * 100,
                           'mean': mean,
                           'label': label}


def get_box(bootstrap_dict):
    return {
        'label': bootstrap_dict['label'],
        'whislo': bootstrap_dict['lower'] / 100,
        'q1': bootstrap_dict['lower'] / 100,
        'med': bootstrap_dict['mean'],
        'q3': bootstrap_dict['upper'] / 100,
        'whishi': bootstrap_dict['upper'] / 100,
        'fliers': []
    }
