import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import timedelta
import seaborn as sns

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'c:/Users/Madjid/Desktop/Mes Cours/Mémoire/Codes/cleaned_weather.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display all column names
print("Columns number:")
print(len(df.columns.tolist()))

print("rows number :")
print(len(df))

print("column names :")
print(df.columns.tolist())

print("first 5 columns of file :")
print(df.head())

time_column = 'date'  
value_columns = ['p', 'T']  
expected_frequency = '1T'  
VALUE_MIN = 100 
VALUE_MAX = 1000 

if time_column not in df.columns:
    raise ValueError(f"Dataset must contain the '{time_column}' column")
for col in value_columns:
    if col not in df.columns:
        raise ValueError(f"Dataset must contain the '{col}' column")

# Ensure that the time column is in datetime format
df[time_column] = pd.to_datetime(df[time_column], errors='coerce')

# Data quality evaluation functions
def evaluate_completeness(df, column):
    total_points = len(df)
    missing_points = df[column].isna().sum()
    completeness_score = (total_points - missing_points) / total_points
    return completeness_score

def evaluate_consistency(df, time_col, value_col):
    duplicates = df.duplicated(subset=[time_col, value_col]).sum()
    consistency_score = 1 - (duplicates / len(df))
    return consistency_score

def evaluate_timeliness(df, time_col, expected_frequency):
    df_sorted = df.sort_values(time_col).dropna(subset=[time_col])
    expected_timestamps = pd.date_range(start=df_sorted[time_col].min(), 
                                        end=df_sorted[time_col].max(), 
                                        freq=expected_frequency)
    actual_timestamps = pd.to_datetime(df_sorted[time_col])
    delay_count = len(expected_timestamps.difference(actual_timestamps))
    timeliness_score = 1 - (delay_count / len(expected_timestamps))
    return timeliness_score

def evaluate_validity(df, column, min_val=VALUE_MIN, max_val=VALUE_MAX):
    invalid_points = df[(df[column] < min_val) | (df[column] > max_val)].shape[0]
    validity_score = 1 - (invalid_points / len(df))
    return validity_score

# Calculate and display data quality scores for each column
quality_scores = {}

for col in value_columns:
    print(f"Evaluating data quality for '{col}':")
    completeness_score = evaluate_completeness(df, col)
    consistency_score = evaluate_consistency(df, time_column, col)
    validity_score = evaluate_validity(df, col)
    timeliness_score = evaluate_timeliness(df, time_column, expected_frequency)

    quality_scores[col] = {
        'Completeness': completeness_score,
        'Consistency': consistency_score,
        'Timeliness': timeliness_score,
        'Validity': validity_score
    }

    # Display data quality scores
    print(f"  Completeness: {completeness_score:.2f}")
    print(f"  Consistency: {consistency_score:.2f}")
    print(f"  Timeliness: {timeliness_score:.2f}")
    print(f"  Validity: {validity_score:.2f}")

# Visualize the quality scores
fig, axes = plt.subplots(1, len(value_columns), figsize=(10 * len(value_columns), 6))

if len(value_columns) == 1:
    axes = [axes]  # Ensure axes is iterable when only one column

for i, col in enumerate(value_columns):
    scores = quality_scores[col]
    axes[i].pie(scores.values(), labels=scores.keys(), autopct='%1.1f%%', startangle=140, colors=cm.coolwarm(np.linspace(0, 1, len(scores))))
    axes[i].set_title(f"Data Quality Score Distribution for '{col}'")

plt.show()