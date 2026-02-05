# %%
import pandas as pd
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling

# %%
# What metrics are indicators of a student graduating?
# Independent Business Metric: Assuming more students graduating equates to more popularity to the college, 
# which factors can be improved to raise the amount of students who graduate.
college_url = ("https://raw.githubusercontent.com/UVADS/DS-3021/refs/heads/main/"
               "data/cc_institution_details.csv")
college = pd.read_csv(college_url)
college.info()

# %%
# What metrics are indicators of someone getting placed to a job?
# Independent Business Metric: Assuming people are constantly searching for jobs, we can search for what indicates
# someone getting placed in a job to help people get jobs.
job_url = ("https://raw.githubusercontent.com/DG1606/CMS-R-2020/"
           "master/Placement_Data_Full_Class.csv")
job = pd.read_csv(job_url)
print(job)
job.info()

# %%
# Looking at what is important for the problem, I am removing gender and salary as they aren't really indicators
# of someone finding employment. Also, I wanted to filter in data purely from the academic field, and by including gender it isn't really relevant.
cols = ["gender", "salary"]
job[cols] = job[cols].astype('category')
job.dtypes

# %%
job.specialisation.value_counts()

# %%
job.ssc_b.value_counts()

# %%
numeric_cols = list(job.select_dtypes('number'))
job[numeric_cols] = MinMaxScaler().fit_transform(job[numeric_cols])



# %%
