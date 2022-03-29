import gzip
import json

import pandas
import glob

from file_anchor import root_dir

languages = [
    'cpp',
    'java',
    'javascript',
    'php',
    'python',
]


def main():
    # prep datasets
    for language in languages:
        stats = {'lang': language}

        df = get_issues_for_language(language)
        df.to_csv(root_dir() + 'datasets/' + language + '_all_issues.csv.zip', compression='zip')
        stats.update({'mined_issues': len(df), 'mined_projects': len(df['projectName'].value_counts())})

        training, validation = training_validation_splits(df)
        validation.to_csv(root_dir() + 'datasets/' + language + '_validation_issues.csv.zip', compression='zip')
        stats.update({'validation_issues': len(validation), 'validation_projects': len(validation['projectName'].value_counts())})

        training.to_csv(root_dir() + 'datasets/' + language + '_training_issues.csv.zip', compression='zip')
        stats.update({'training_issues': len(training), 'training_projects': len(training['projectName'].value_counts())})

        create_manual_classification_files(validation, language)

        md_df = get_documentation_for_language(language)
        md_df.to_csv(root_dir() + 'datasets/' + language + '_all_documentation.csv.zip', compression='zip')
        stats.update({'documentation_files': len(md_df), 'documentation_projects': len(md_df['projectName'].value_counts())})

        with open(root_dir() + 'datasets/' + language + '_stats.json', 'w') as fd:
            json.dump(stats, fd)

    # get manual validation set
    for language in languages:
        print(language)
        researcher_1 = collect_manual_validation_set(root_dir() + 'githubMiner/issue_tickets_researcher_1/' + language + '/')
        researcher_1.to_csv(root_dir() + 'datasets/' + language + '_reseracher_1_manually_labeled_validation_set.csv.zip', compression='zip')
        researcher_2 = collect_manual_validation_set(root_dir() + 'githubMiner/issue_tickets_researcher_2/' + language + '/')
        researcher_2.to_csv(root_dir() + 'datasets/' + language + '_reseracher_2_manually_labeled_validation_set.csv.zip', compression='zip')

    # get link list for validation set:
    for language in languages:
        df = pandas.read_csv(root_dir() + 'datasets/' + language + '_validation_issues.csv.zip', compression='zip')
        df['url'].to_csv(root_dir() + 'datasets/' + language + '_validation_issues_link_list.csv')


def create_manual_classification_files(df, language):
    df['title'] = df['title'].fillna('')
    df['body'] = df['body'].fillna('')
    for index, issue in df.iterrows():
        issue_df = pandas.DataFrame()
        issue_df['target'] = ''
        issue_df['doc'] = (issue['title'] + '\n' + issue['body']).splitlines()

        file_name = issue['url'].replace('https://api.github.com/repos/', '').replace('/', '_') + '.csv'
        issue_df.to_csv(root_dir() + 'datasets/' + language + '/' + file_name)


def training_validation_splits(df):
    validation = df.sample(250, random_state=42)
    rest = df.drop(validation.index)
    training = rest[rest['body'].str.contains("```", na=False)]
    return training, validation


def get_issues_for_language(language):
    path = root_dir() + 'githubMiner/json_dump/' + language + '/'
    issues_df = pandas.DataFrame()
    for json_gz_path in glob.glob(path + '*.json.gz'):
        if json_gz_path.endswith('_documentation.json.gz'):
            continue
        with gzip.open(json_gz_path, mode='rt', encoding="utf-8") as fd:
            issues_json = json.load(fd)

        df = pandas.DataFrame([{'body': x['body'],
                                'title': x['title'],
                                'url': x['url'],
                                'projectName': x['url'].replace('https://api.github.com/repos/', '').split('/issues/')[0]}
                               for x in issues_json])
        issues_df = issues_df.append(df)
    issues_df = issues_df.reset_index()
    return issues_df


def get_documentation_for_language(language):
    path = root_dir() + 'githubMiner/json_dump/' + language + '/'
    docs_df = pandas.DataFrame()
    for json_gz_path in glob.glob(path + '*_documentation.json.gz'):
        with gzip.open(json_gz_path, mode='rt', encoding="utf-8") as fd:
            docs_json = json.load(fd)

        markdown_docs = []
        for doc in docs_json:
            if isinstance(doc, dict) and \
                    'decoded_content' in doc.keys() and \
                    (doc['name'].endswith('.md') or doc['name'].endswith('.markdown')):
                markdown_docs.append({'url': doc['url'],
                                      'path': doc['path'],
                                      'projectName': doc['url'].replace('https://api.github.com/repos/', '').split('/contents/')[0],
                                      'content': doc['decoded_content']})
        docs_df = docs_df.append(pandas.DataFrame(markdown_docs))
    return docs_df


def collect_manual_validation_set(path):
    df = pandas.DataFrame()
    for csv_file in glob.glob(path + '*.csv'):
        bug_df = pandas.read_csv(csv_file)
        bug_df['origin'] = csv_file.split('/')[-1].replace('.csv', '')
        df = df.append(bug_df, ignore_index=True)
    return df


if __name__ == '__main__':
    main()
