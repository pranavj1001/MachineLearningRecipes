# Your code goes here :)

# import package with helper functions
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

# print couple of rows from the table accident_2015
accidents.head("accident_2015")

# query for the first question
query_for_first_question = """SELECT COUNT(consecutive_number),
                              EXTRACT(HOUR FROM timestamp_of_crash)
                              FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                              GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
                              ORDER BY COUNT(consecutive_number) DESC
                           """

# run the query
accidents_by_hour = accidents.query_to_pandas_safe(query_for_first_question)

# print the dataframe
accidents_by_hour

# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")


# print couple of rows from the table vehicle_2015
accidents.head("vehicle_2015")

# query for the second question
query_for_second_question = """
                               SELECT registration_state_name,
                               COUNTIF(hit_and_run = 'Yes')
                               FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                               GROUP BY registration_state_name
                               ORDER BY COUNTIF(hit_and_run = 'Yes') DESC
                            """

# run the query
hit_and_runs = accidents.query_to_pandas_safe(query_for_second_question)

# print the dataframe
hit_and_runs
