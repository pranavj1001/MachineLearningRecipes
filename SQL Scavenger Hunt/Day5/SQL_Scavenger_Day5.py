# Your code goes here :)
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")

# print a couple of rows of the sample_commits table
github.head("sample_commits")

# print a couple of rows of the sample_files table
github.head("sample_files")

query_for_first_question = """WITH py AS (
                                  SELECT DISTINCT repo_name
                                  FROM `bigquery-public-data.github_repos.sample_files`
                                  WHERE path LIKE '%.py')
                              SELECT c.repo_name, COUNT(commit) AS number_of_commits
                              FROM `bigquery-public-data.github_repos.sample_commits` AS c
                              JOIN py ON  py.repo_name = c.repo_name
                              GROUP BY c.repo_name
                              ORDER BY number_of_commits DESC
                           """

# check the size of the query
github.estimate_query_size(query_for_first_question)

# run the query
number_of_py_commits = github.query_to_pandas_safe(query_for_first_question, max_gb_scanned=6)

# print the result
number_of_py_commits
