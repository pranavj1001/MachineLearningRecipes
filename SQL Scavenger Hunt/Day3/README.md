# SQL Scanvenger Hunt Day 3

Hey there,

The questions for Day3 are:

1) Which hours of the day do the most accidents occur during?

* Return a table that has information on how many accidents occurred in each hour of the day in 2015, sorted by the the number of accidents which occurred each hour. Use either the accident2015 or accident2016 table for this, and the timestampofcrash column. (Yes, there is an hourofcrash column, but if you use that one you won't get a chance to practice with dates. :P)
**Hint**: You will probably want to use the EXTRACT() function for this.

2) Which state has the most hit and runs?

* Return a table with the number of vehicles registered in each state that were involved in hit-and-run accidents, sorted by the number of hit and runs. Use either the vehicle2015 or vehicle2016 table for this, especially the registrationstatename and hitandrun columns.

**Dataset**: [US Traffic Fatality Records](https://www.kaggle.com/usdot/nhtsa-traffic-fatalities)