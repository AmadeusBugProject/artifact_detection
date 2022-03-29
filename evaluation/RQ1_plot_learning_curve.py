import pandas
from matplotlib import pyplot as plt

from artifact_detection_model.utils.Logger import Logger
from datasets.constants import LANGUAGES

from file_anchor import root_dir

log = Logger()

OUT_PATH = root_dir() + 'evaluation/out/learning_curve/'

language_labels = {
    'cpp': 'C++',
    'java': 'Java',
    'javascript': 'JavaScript',
    'php': 'PHP',
    'python': 'Python',
}


def main():
    fig = plt.figure(figsize=(8, 10))
    axes = []
    axes.append(fig.add_subplot(3, 2, 1))
    axes.append(fig.add_subplot(3, 2, 2, sharex=axes[0], sharey=axes[0]))
    axes.append(fig.add_subplot(3, 2, 3, sharex=axes[0], sharey=axes[0]))
    axes.append(fig.add_subplot(3, 2, 4, sharex=axes[0], sharey=axes[0]))
    axes.append(fig.add_subplot(3, 2, 5, sharex=axes[0], sharey=axes[0]))
    axes.append(fig.add_subplot(3, 2, 6, sharex=axes[0]))

    for i, lang in enumerate(LANGUAGES):
        print(lang)
        df = pandas.read_csv(OUT_PATH + lang + '_artifact_detection_summary.csv')
        df = df[df['train_samples'] <= 800000]
        df['train_samples'] = df['train_samples']/ 10**3
        ax = axes[i]
        gb = df.groupby(by='train_samples')

        plot_mean_and_fill_std(ax, gb, 'roc-auc_' + lang + '_researcher_1', 'g', 'Validation set 1', style='o-')
        plot_mean_and_fill_std(ax, gb, 'roc-auc_' + lang + '_researcher_2', 'b', 'Validation set 2', style='v-')
        ax.title.set_text(language_labels[lang])

        colors = ['red', 'purple', 'violet', 'k', 'c']
        styles = ['p-', '*-', 'v-', 'D-', 'X-']
        plot_mean_and_fill_std(axes[5], gb, 'model_size', colors[i], language_labels[lang], style=styles[i])

    axes[0].set_ylabel('ROC-AUC')
    axes[2].set_ylabel('ROC-AUC')
    axes[4].set_ylabel('ROC-AUC')
    axes[4].set_xlabel('Training set size (10^3 lines)')
    axes[5].set_xlabel('Training set size (10^3 lines)')
    axes[0].legend(loc='lower right')

    axes[5].set_ylabel('Model size (MiB)')
    axes[5].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(OUT_PATH + 'roc-auc_validation_set_learning_curve.pdf')


def plot_mean_and_fill_std(axes, gb, metric, color, label, style='o-'):
    axes.fill_between(gb.mean().index, gb.mean()[metric] - gb.std()[metric],
                         gb.mean()[metric] + gb.std()[metric], alpha=0.1,
                         color=color)
    axes.plot(gb.mean().index, gb.mean()[metric], style, color=color, label=label)


if __name__ == "__main__":
    main()
