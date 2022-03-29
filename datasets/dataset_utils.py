from os.path import dirname

import pandas

from artifact_detection_model.constants import TARGET_NAMES
from artifact_detection_model.regex_cleanup import split_by_md_code_block, regex_cleanup
from datasets.constants import LANGUAGES
from file_anchor import root_dir

def get_validation_sets_for_language(lang):
    r_1_val_set = pandas.read_csv(
        root_dir() + 'datasets/' + lang + '_reseracher_1_manually_labeled_validation_set.csv.zip', compression='zip')
    r_1_val_set = r_1_val_set[~r_1_val_set['target'].isnull()]
    r_1_val_set = r_1_val_set[~r_1_val_set['doc'].isnull()]

    r_2_val_set = pandas.read_csv(
        root_dir() + 'datasets/' + lang + '_reseracher_2_manually_labeled_validation_set.csv.zip', compression='zip')
    r_2_val_set = r_2_val_set[~r_2_val_set['target'].isnull()]
    r_2_val_set = r_2_val_set[~r_2_val_set['doc'].isnull()]

    return {lang + '_researcher_1': r_1_val_set, lang + '_researcher_2': r_2_val_set}

def get_all_validation_sets():
    validation_sets = {}
    for lang in LANGUAGES:
        validation_sets.update(get_validation_sets_for_language(lang))
    return validation_sets


def get_data_from_issues(df, regex_clean=True):
    df = df[df['body'].str.contains("```", na=False)]
    df['body'] = df['body'].fillna('')
    df['title'] = df['title'].fillna('')
    docs = df['title'] + '\n' + df['body']
    documents = docs.tolist()

    artifacts, text = split_by_md_code_block(documents)

    if regex_clean:
        art, text = regex_cleanup(text)
        artifacts.extend(art)

    return artifacts, text


def get_data_from_documentation(df, regex_clean=True):
    df = df[~df['content'].isnull()].copy()
    df['doc'] = df['content'].astype(str)
    documents = df.pop('doc').values

    artifacts, text = split_by_md_code_block(documents)

    if regex_clean:
        art, text = regex_cleanup(text)
        artifacts.extend(art)

    return artifacts, text


def get_trainingset(lang, balance=True):
    df = pandas.read_csv(root_dir() + 'datasets/' + lang + '_training_issues.csv.zip', compression='zip')
    issue_artifacts, issue_nat_lang = get_data_from_issues(df)

    df = pandas.read_csv(root_dir() + 'datasets/' + lang + '_all_documentation.csv.zip', compression='zip')
    documentation_artifacts, documentation_nat_lang = get_data_from_documentation(df)

    df_nat_lang = pandas.DataFrame({'doc': issue_nat_lang + documentation_nat_lang})
    df_nat_lang['target'] = TARGET_NAMES['text']
    df_artifacts = pandas.DataFrame({'doc': issue_artifacts + documentation_artifacts})
    df_artifacts['target'] = TARGET_NAMES['artifact']

    if balance:
        df_train = df_nat_lang.append(df_artifacts.sample(len(df_nat_lang), random_state=42))
    else:
        df_train = df_nat_lang.append(df_artifacts)
    return df_train