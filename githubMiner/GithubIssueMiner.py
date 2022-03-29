import gzip
import time
import json
import copy

from github import Github  # pygithub
from github import RateLimitExceededException # when rate limit is hit
from github import UnknownObjectException # when event or issue id is unknown


def rate_limit_handler(function):
    def wrapper(*args, **kwargs):
        while True:
            try:
                return function(*args, **kwargs)
            except RateLimitExceededException:
                print("RateLimitExceeded - waiting")
                print("Currently at: " + function.__name__)
                print("Current time: " + time.ctime())
                time.sleep(4000)
    return wrapper


class GithubIssueMiner:
    def __init__(self, github_api_key, project_name, out_path, skip_pull_requests=True, skip_event_timeline=True, skip_event_types=None):
        self.github_api_key = github_api_key
        self.github = Github(github_api_key)
        self.project_name = project_name
        self.project = self.__init_github_project(project_name)
        self.skip_pull_requests = skip_pull_requests
        self.skip_event_timeline = skip_event_timeline
        self.skip_event_types = skip_event_types
        self.out_path = out_path
        self.db = []

    @rate_limit_handler
    def __init_github_project(self, project_name):
        return self.github.get_repo(project_name)

    @rate_limit_handler
    def __get_issue(self, issues_plist, issue_index):
        return issues_plist[issue_index]

    @rate_limit_handler
    def __get_issues_list(self):
        return self.project.get_issues(state='closed', direction='asc')

    @rate_limit_handler
    def __get_event_timeline(self, issue):
        return issue.get_timeline()

    @rate_limit_handler
    def __get_event(self, event_plist, event_index):
        return event_plist[event_index]

    @rate_limit_handler
    def __get_raw_data(self, item):
        return item.raw_data

    def __to_db(self, dict_item):
        # data = json.dumps(item.raw_data)
        self.db.append(dict_item)

    def __fetch_events(self, issue):
        timeline_url = issue.raw_data['timeline_url']
        events_plist = self.__get_event_timeline(issue)
        for event_index in range(0, events_plist.totalCount):
            event = self.__get_event(events_plist, event_index)
            raw_event = copy.deepcopy(self.__get_raw_data(event))
            if self.skip_event_types and raw_event['event'] in self.skip_event_types:
                continue
            raw_event.update({'timeline_url': timeline_url})
            self.__to_db(raw_event)

    def fetch(self):
        issues_plist = self.__get_issues_list()
        for issue_index in range(0, issues_plist.totalCount):
            issue = self.__get_issue(issues_plist, issue_index)
            raw_issue = self.__get_raw_data(issue)
            if self.skip_pull_requests and 'pull_request' in raw_issue.keys():
                continue
            self.__to_db(raw_issue)

            if not self.skip_event_timeline:
                self.__fetch_events(issue)

            print(issue.html_url + " , nr. " + str(issue_index) + " done! " + str(issues_plist.totalCount - issue_index - 1) + " issues to go...")
        print('Done')

    def save_data(self):
        path = self.out_path + self.project_name.replace('/', '_') + '.json.gz'
        with gzip.open(path, mode='wt', encoding="utf-8") as fd:
            json.dump(self.db, fd)
