# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling

"""
Step 1
"""

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

"""
Step 2
"""

"""
College
"""

# %%
# Create target variable, try to separate to a high and low graduation rates to classify schools
college = college.loc[college["grad_100_value"].notna()].copy()
college["high_grad_4y"] = (college["grad_100_value"] >= 50).astype(int)

print("Target distribution:")
print(college["high_grad_4y"].value_counts())
college_prev = (college.high_grad_4y.value_counts()[1] / len(college.high_grad_4y))
print(f"Prevalence: {college_prev:.2%}")

# Manual Calculation
num_high_grad = 690
total = (690+2777)
print(num_high_grad / total)

# %%
# Drop identifiers as they are irrelevant to our model
college = college.drop(["index", "unitid", "chronname", "city", "site", "nicknames", "similar"], axis=1)
missing = college.isna().mean()
drop_cols = missing[missing > .5].index.tolist()
college = college.drop(columns=drop_cols)

# %%
# Since there is a lot of missing data in this data set, I am going to populate numeric columns with median and categorical/object columns with mode
num_cols = list(college.select_dtypes(include=[np.number]).columns)
for col in num_cols:
    if college[col].isna().any():
        college[col] = college[col].fillna(college[col].median())

obj_cols = list(college.select_dtypes(include=["object", "category"]).columns)
for col in obj_cols:
    if college[col].isna().any():
        mode = college[col].mode(dropna=True)
        fill = mode.iloc[0] if len(mode) else "Unknown"
        college[col] = college[col].fillna(fill)

# Correctly label state
college["state"] = college["state"].astype("category")

print(college.state.value_counts())

# Collapsing state into top 5 and then the rest other to condense
top_states = ['California', 'New York', 'Pennsylvania', 'Texas', 'Ohio']
college.state = (college.state.apply(lambda x: x if x in top_states
                                     else "Other")).astype('category')

print(college.state.value_counts())

# %%
# Scaling numeric columns so the scale does not interfere
numeric_cols = list(college.select_dtypes("number").columns)
numeric_cols = [c for c in numeric_cols if c != "high_grad_4yr"]
college[numeric_cols] = MinMaxScaler().fit_transform(college[numeric_cols])

# %%
# Applying One-hot encoding
categories = list(college.select_dtypes("category"))
college_clean = pd.get_dummies(college, columns=categories)


# %%
# Splitting
train_c, test_c = train_test_split(
    college_clean,
    stratify=college_clean.high_grad_4y
)

# %%
# Verifying the split sizes
print(f"Training set shape: {train_c.shape}")
print(f"Test set shape: {test_c.shape}")

# %%
# Second split: Split remaining data into tuning and test sets (50/50)
tune_c, test_c = train_test_split(
    test_c,
    train_size=.5,
    stratify=test_c.high_grad_4y
)

# %%
# Verify prevalence in training set
print("Training set class distribution:")
print(train_c.high_grad_4y.value_counts())
train_counts = train_c["high_grad_4y"].value_counts()
print(f"Training prevalence: {train_counts[1]}/{train_counts.sum()} = {train_counts[1]/train_counts.sum():.2%}")

# %%
# Verify prevalence in tuning set
print("Tuning set class distribution:")
print(tune_c.high_grad_4y.value_counts())
tune_counts = tune_c["high_grad_4y"].value_counts()
print(f"Tuning prevalence: {tune_counts[1]}/{tune_counts.sum()} = {tune_counts[1]/tune_counts.sum():.2%}")

# %%
# Verify prevalence in test set
print("Test set class distribution:")
print(test_c.high_grad_4y.value_counts())
test_counts = test_c["high_grad_4y"].value_counts()
print(f"Test prevalence: {test_counts[1]}/{test_counts.sum()} = {test_counts[1]/test_counts.sum():.2%}")

"""
Job
"""
# %%
job.specialisation.value_counts()

# %%
job.ssc_b.value_counts()

# %%
# Create target variable and calculate prevalence
# Setting placed or not placed to 1's and 0's
job["placed"] = (job["status"] == "Placed").astype(int)

print("Target distribution:")
print(job["placed"].value_counts())
job_prev = (job.placed.value_counts()[1] / len(job.placed))
print(f"Prevalence: {job_prev:.2%}")

# %%
# Looking at what is important for the problem, I am removing the sl_no, status, gender and salary as they aren't really indicators
# of someone finding employment. Also, I wanted to filter in data purely from the academic field, and by including gender it isn't really relevant.
cols = ["sl_no", "status", "gender", "salary"]
job = job.drop(cols, axis=1)
job.dtypes

# %%
# Remove rows with any missing values
# notna() returns True for non-missing values
# all(axis='columns') checks if all values in a row are non-missing
job_clean = job.loc[job.notna().all(axis='columns')]

# %%
# Summarize missing values in the job dataset
# isna() returns True for missing values, sum() counts them
print(job.isna().sum())

# %%
# Correct variable types
cols = ["ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation"]
job[cols] = job[cols].astype("category")
print(job.dtypes)

# %%
# Select numeric columns and scale
numeric_cols = list(job.select_dtypes('number').columns)
numeric_cols = [c for c in numeric_cols if c != "placed"]

job[numeric_cols] = MinMaxScaler().fit_transform(job[numeric_cols])

# One-hot encoding
category_list = list(job.select_dtypes("category"))
job_clean = pd.get_dummies(job, columns=category_list)

# %%
# Splitting
train_j, test_j = train_test_split(
    job_clean,
    stratify=job.placed
)

# %%
# Verifying the split sizes
print(f"Training set shape: {train_j.shape}")
print(f"Test set shape: {test_j.shape}")

# %%
# Second split: Split remaining data into tuning and test sets (50/50)
tune_j, test_j = train_test_split(
    test_j,
    train_size=.5,
    stratify=test_j.placed
)

# %%
# Verify prevalence in training set
print("Training set class distribution:")
print(train_j.placed.value_counts())
train_counts = train_j["placed"].value_counts()
print(f"Training prevalence: {train_counts[1]}/{train_counts.sum()} = {train_counts[1]/train_counts.sum():.2%}")

# %%
# Verify prevalence in tuning set
print("Tuning set class distribution:")
print(tune_j.placed.value_counts())
tune_counts = tune_j["placed"].value_counts()
print(f"Tuning prevalence: {tune_counts[1]}/{tune_counts.sum()} = {tune_counts[1]/tune_counts.sum():.2%}")

# %%
# Verify prevalence in test set
print("Test set class distribution:")
print(test_j.placed.value_counts())
test_counts = test_j["placed"].value_counts()
print(f"Test prevalence: {test_counts[1]}/{test_counts.sum()} = {test_counts[1]/test_counts.sum():.2%}")


"""
Step Three
What do your instincts tell you about the data. Can it address your problem, what areas/items are you worried about?
"""

# College Data Set:
# I think this data can address my problem, however it doesn't track individual students but different institutions. Because of this, there may be a lot of noise within training.
# Also, with missing information from different states, information may be skewed by how well data is recorded in different parts of the U.S.
# The other graduation data may affect the 4 year graduation target by fitting close to the target.

# Job Data Set:
# The job data set tells us information that we can definitely use. I did drop salary as it doesn't affect our goal, job placement.
# Although, the data set is relatively small so there is a worry of the model overfitting.
# I dropped gender as I worried about bias.
# There isn't data on how well people performed in interviews, etc.
