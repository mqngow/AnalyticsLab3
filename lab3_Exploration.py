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

# %%
# Create target variable, try to separate to a high and low graduation rates to classify schools
college = college.loc[college["grad_100_value"].notna()].copy()
college["high_grad_4y"] = (college["grad_100_value"] >= 50).astype(int)

print("\nTarget distribution:")
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
train, test = train_test_split(
    college_clean,
    stratify=college_clean.high_grad_4y
)

# %%
# Verifying the split sizes
print(f"Training set shape: {train.shape}")
print(f"Test set shape: {test.shape}")

# %%
# Second split: Split remaining data into tuning and test sets (50/50)
tune, test = train_test_split(
    test,
    train_size=.5,
    stratify=test.high_grad_4y
)

# %%
# Verify prevalence in training set
print("Training set class distribution:")
print(train.high_grad_4y.value_counts())
train_counts = train["high_grad_4y"].value_counts()
print(f"Training prevalence: {train_counts[1]}/{train_counts.sum()} = {train_counts[1]/train_counts.sum():.2%}")

# %%
# Verify prevalence in tuning set
print("Tuning set class distribution:")
print(tune.high_grad_4y.value_counts())
tune_counts = tune["high_grad_4y"].value_counts()
print(f"Tuning prevalence: {tune_counts[1]}/{tune_counts.sum()} = {tune_counts[1]/tune_counts.sum():.2%}")

# %%
# Verify prevalence in test set
print("Test set class distribution:")
print(test.high_grad_4y.value_counts())
test_counts = test["high_grad_4y"].value_counts()
print(f"Test prevalence: {test_counts[1]}/{test_counts.sum()} = {test_counts[1]/test_counts.sum():.2%}")

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

"""
Step Three
"""
