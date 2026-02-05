"""
Step 4
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Calculate prevalence.
def calculate_prevalence(y: pd.Series) -> float:
    counts = y.value_counts()
    return counts[1] / len(y)

# Drop columns.
def drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
	return df.drop(cols, axis=1)

# Drop columns with many missing values.
def drop_high_missing_cols(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
	missing = df.isna().mean()
	drop_cols = missing[missing > threshold].index.tolist()
	return df.drop(columns=drop_cols)

# Consolidating top levels.
def collapse_top_levels(df: pd.DataFrame, col: str, top_levels: list[str], other_label: str = "Other") -> pd.DataFrame:
	if col in df.columns:
		df[col] = df[col].apply(lambda x: x if x in top_levels else other_label).astype("category")
	return df

# Min Max Scaling.
def scale_numeric_columns(df: pd.DataFrame, exclude_cols: list[str] = None) -> pd.DataFrame:
	df = df.copy()
	if exclude_cols is None:
		exclude_cols = []
	
	numeric_cols = list(df.select_dtypes("number").columns)
	numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
	
	if numeric_cols:
		df[numeric_cols] = MinMaxScaler().fit_transform(df[numeric_cols])
	
	return df

# One Hot Encoding.
def one_hot_encode(df: pd.DataFrame, categorical_cols: list[str] = None) -> pd.DataFrame:
	if categorical_cols is None:
		categorical_cols = list(df.select_dtypes("category").columns)
	
	return pd.get_dummies(df, columns=categorical_cols)

# Splitting Train Test and Tune.
def split_train_test_tune(df: pd.DataFrame, target_col: str, test_size: float = 0.25) -> tuple:
	# First split: train and temp.
	train, test = train_test_split(
		df,
		test_size=test_size,
		stratify=df[target_col]
	)
	
	# Second split: split temp into tune and test (50/50).
	tune, test = train_test_split(
		test,
		train_size=0.5,
		stratify=test[target_col]
	)
	
	return train, tune, test

# Print out split sizes and prevalence.
def print_split_prevalence(train: pd.DataFrame, tune: pd.DataFrame, test: pd.DataFrame, target_col: str) -> None:
	print("Split sizes:")
	print(f"Train: {train.shape}")
	print(f"Tune:  {tune.shape}")
	print(f"Test:  {test.shape}")
	
	print("Training set class distribution:")
	print(train[target_col].value_counts())
	train_counts = train[target_col].value_counts()
	print(f"Training prevalence: {train_counts[1]}/{train_counts.sum()} = {train_counts[1]/train_counts.sum():.2%}")
	
	print("Tuning set class distribution:")
	print(tune[target_col].value_counts())
	tune_counts = tune[target_col].value_counts()
	print(f"Tuning prevalence: {tune_counts[1]}/{tune_counts.sum()} = {tune_counts[1]/tune_counts.sum():.2%}")
	
	print("Test set class distribution:")
	print(test[target_col].value_counts())
	test_counts = test[target_col].value_counts()
	print(f"Test prevalence: {test_counts[1]}/{test_counts.sum()} = {test_counts[1]/test_counts.sum():.2%}")
