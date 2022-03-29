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


class GithubDocumentationMiner:
    def __init__(self, github_api_key, project_name, out_path):
        self.github_api_key = github_api_key
        self.github = Github(github_api_key)
        self.project_name = project_name
        self.out_path = out_path
        self.db = []

    @rate_limit_handler
    def __get_content_file(self, file_plist, file_index):
        return file_plist[file_index]

    @rate_limit_handler
    def __get_file_list(self, query):
        return self.github.search_code(query)

    @rate_limit_handler
    def __get_paginated_list_total_count(self, plist):
        return plist.totalCount

    @rate_limit_handler
    def __get_raw_data(self, item):
        return item.raw_data

    def __to_db(self, dict_item):
        # data = json.dumps(item.raw_data)
        self.db.append(dict_item)

    def fetch(self):
        self.__fetch_files_by_query('repo:' + self.project_name + ' .md in:path')
        self.__fetch_files_by_query('repo:' + self.project_name + ' .markdown in:path')

    def __fetch_files_by_query(self, query):
        file_count = 0
        file_plist = []
        for _ in range(0, 3):
            file_plist = self.__get_file_list(query)
            file_count = self.__get_paginated_list_total_count(file_plist)
            if file_count > 0:
                break
        file_index = 0
        while file_index < file_count:
            content_file = None

            for _ in range(0, 2):
                try:
                    content_file = self.__get_content_file(file_plist, file_index)
                    break
                except IndexError:
                    file_plist = self.__get_file_list(query)
                    file_count = self.__get_paginated_list_total_count(file_plist)

            if content_file:
                raw_c_file = copy.deepcopy(self.__get_raw_data(content_file))
                try:
                    raw_c_file['decoded_content'] = content_file.decoded_content.decode('UTF-8')
                except UnicodeDecodeError:
                    print('UnicodeDecodeError at ' + query + ' file: ' + raw_c_file["path"])
                self.__to_db(raw_c_file)
            else:
                self.__to_db({'failed file': self.project_name})

            file_index += 1

    def save_data(self):
        path = self.out_path + self.project_name.replace('/', '_') + '_documentation.json.gz'
        with gzip.open(path, mode='wt', encoding="utf-8") as fd:
            json.dump(self.db, fd)
