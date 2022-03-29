import glob
import json
import traceback

from GithubDocumentationMiner import GithubDocumentationMiner
from GithubIssueMiner import GithubIssueMiner
from artifact_detection_model.utils.Logger import Logger

from file_anchor import root_dir

log = Logger(log_file='log.txt')

skip_event_types = [
                    'commit-commented',
                    'line-commented',
                    'unsubscribed',
                    'marked_as_duplicate',
                    'ready_for_review',
                    'head_ref_restored',
                    'head_ref_deleted',
                    'renamed',
                    'review_dismissed',
                    'head_ref_force_pushed',
                    'locked',
                    'convert_to_draft',
                    'base_ref_force_pushed',
                    'milestoned',
                    'subscribed',
                    'labeled',
                    'mentioned',
                    'referenced',
                    # 'committed',
                    # 'closed',
                    'cross-referenced',
                    'demilestoned',
                    'reopened',
                    'unlabeled',
                    'review_requested',
                    'merged',
                    'assigned',
                    # 'commented',
                    'reviewed'
]


def main():
    with open(root_dir() + '../github_api_key.json', 'r') as fd:
        github_api_key = json.load(fd)['github_api_key']
    # github_api_key = None

    cpp_projects = load_project_list(root_dir() + 'githubMiner/json_dump/cpp.txt')
    python_projects = load_project_list(root_dir() + 'githubMiner/json_dump/python.txt')
    java_projects = load_project_list(root_dir() + 'githubMiner/json_dump/java.txt')
    javascript_projects = load_project_list(root_dir() + 'githubMiner/json_dump/javascript.txt')
    php_projects = load_project_list(root_dir() + 'githubMiner/json_dump/php.txt')

    fetch_all_issues(github_api_key, cpp_projects, root_dir() + 'githubMiner/json_dump/cpp/')
    fetch_all_issues(github_api_key, python_projects, root_dir() + 'githubMiner/json_dump/python/')
    fetch_all_issues(github_api_key, java_projects, root_dir() + 'githubMiner/json_dump/java/')
    fetch_all_issues(github_api_key, javascript_projects, root_dir() + 'githubMiner/json_dump/javascript/')
    fetch_all_issues(github_api_key, php_projects, root_dir() + 'githubMiner/json_dump/php/')

    fetch_all_documentation(github_api_key, cpp_projects, root_dir() + 'githubMiner/json_dump/cpp/')
    fetch_all_documentation(github_api_key, python_projects, root_dir() + 'githubMiner/json_dump/python/')
    fetch_all_documentation(github_api_key, java_projects, root_dir() + 'githubMiner/json_dump/java/')
    fetch_all_documentation(github_api_key, javascript_projects, root_dir() + 'githubMiner/json_dump/javascript/')
    fetch_all_documentation(github_api_key, php_projects, root_dir() + 'githubMiner/json_dump/php/')


def fetch_all_issues(github_api_key, projects_list, out_path):
    for project in projects_list:
        if project.replace('/', '_') in [x.replace(out_path, '').replace('.json.gz', '') for x in glob.glob(out_path + '*.json.gz')]:
            log.s('skipping issues' + project)
            continue
        try:
            ghif = GithubIssueMiner(github_api_key, project, out_path)
            ghif.fetch()
            ghif.save_data()
            log.s('done with issues ' + project)
        except Exception as e:
            log.e('issues exception at ' + project)
            log.e(str(e))
            log.e(str(traceback.format_tb(e.__traceback__)))


def fetch_all_documentation(github_api_key, projects_list, out_path):
    for project in projects_list:
        if project.replace('/', '_') in [x.replace(out_path, '').replace('_documentation.json.gz', '') for x in glob.glob(out_path + '*_documentation.json.gz')]:
            log.s('skipping documentation ' + project)
            continue
        try:
            ghdm = GithubDocumentationMiner(github_api_key, project, out_path)
            ghdm.fetch()
            if len(ghdm.db):
                ghdm.save_data()
                log.s('done with documentation ' + project)
            else:
                log.s('empty documentation ' + project)
        except Exception as e:
            log.e('documentation exception at ' + project)
            log.e(str(e))
            log.e(str(traceback.format_tb(e.__traceback__)))


def load_project_list(list_file):
    with open(list_file, 'r') as fd:
        urls = fd.read().splitlines()
    return [x.replace('https://github.com/', '').strip('/') for x in urls]


if __name__ == '__main__':
    main()

