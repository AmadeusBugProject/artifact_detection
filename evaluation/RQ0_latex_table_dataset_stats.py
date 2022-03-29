import pandas

from artifact_detection_model.utils.Logger import Logger
from datasets.constants import LANGUAGES
from datasets.dataset_utils import get_validation_sets_for_language, \
    get_data_from_issues, get_data_from_documentation
from evaluation import RQ0_interrater_agreement
from file_anchor import root_dir

log = Logger()

OUT_PATH = root_dir() + 'evaluation/out/dataset_stats/'

language_labels = {
    'cpp': 'C++',
    'java': 'Java',
    'javascript': 'JavaScript',
    'php': 'PHP',
    'python': 'Python',
}


def main():
    RQ0_interrater_agreement.main()
    reports = []
    for lang in LANGUAGES:
        report = get_dataset_stats(lang)
        print(report)
        reports.append(report)

    df = pandas.DataFrame(reports)

    df = df.set_index('language').T
    df.to_csv(OUT_PATH + 'dataset_stats.csv')
    df.to_latex(OUT_PATH + 'dataset_stats.tex', float_format="%.2f")


def get_dataset_stats(lang):
    report = {'language': language_labels[lang]}
    df = pandas.read_csv(root_dir() + 'datasets/' + lang + '_all_issues.csv.zip', compression='zip')
    report.update({'Number of issues': len(df)})
    report.update({'Issues containing MD codeblocks': len(df[df['body'].str.contains("```", na=False)])})

    df = pandas.read_csv(root_dir() + 'datasets/' + lang + '_training_issues.csv.zip', compression='zip')
    issue_artifacts, issue_nat_lang = get_data_from_issues(df)
    report.update({'Issues in training set': len(df)})

    report.update({'Artifact lines from issues': len(issue_artifacts)})
    report.update({'Natural language lines from issues': len(issue_nat_lang)})

    df = pandas.read_csv(root_dir() + 'datasets/' + lang + '_all_documentation.csv.zip', compression='zip')
    report.update({'Number of documentation files': len(df)})
    documentation_artifacts, documentation_nat_lang = get_data_from_documentation(df)
    report.update({'Artifact lines from documentation': len(documentation_artifacts)})
    report.update({'Natural language lines from documentation': len(documentation_nat_lang)})

    report.update({'Lines in full training set': len(issue_artifacts) + len(issue_nat_lang) + len(documentation_artifacts) + len(documentation_nat_lang)})
    report.update({'Artifact lines in full training set': len(issue_artifacts) + len(documentation_artifacts)})
    report.update({'Natural language lines in full training set': len(issue_nat_lang) + len(documentation_nat_lang)})

    report.update({'Number of issues in validation set': 250})
    val_sets = get_validation_sets_for_language(lang)
    val1_df = val_sets[ lang + '_researcher_1']
    report.update({'Artifact lines in validation set 1': len(val1_df[val1_df['target'] == 0])})
    report.update({'Natural language lines in validation set 1': len(val1_df[val1_df['target'] == 1])})

    val2_df = val_sets[ lang + '_researcher_2']
    report.update({'Artifact lines in validation set 2': len(val2_df[val2_df['target'] == 0])})
    report.update({'Natural language lines in validation set 2': len(val2_df[val2_df['target'] == 1])})

    df = pandas.read_csv(root_dir() + 'evaluation/out/interrater_agreement/' + lang + '_reviewer1_vs_reviewer2_manual_agreement.csv')
    report.update({'Cohens Kappa': df['cohens_kappa'].values[0]})
    report.update({'ROC-AUC': df['roc_auc'].values[0]})
    return report


if __name__ == "__main__":
    main()
