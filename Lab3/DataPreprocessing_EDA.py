# Lab 3 - Data Preprocessing, EDA, NumPy, Pandas and Visualization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# --------------------------------------------------
# 1. NUMPY
# --------------------------------------------------

print("===== NUMPY =====")

numbers = np.array([10, 20, 30, 40, 50])

print("Array:", numbers)
print("Mean:", np.mean(numbers))
print("Standard Deviation:", np.std(numbers))
print("Maximum:", np.max(numbers))
print("Minimum:", np.min(numbers))


# --------------------------------------------------
# 2. LOAD DATASET USING PANDAS
# --------------------------------------------------

print("\n===== PANDAS =====")

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["target"] = iris.target

print("\nFirst 5 rows:")
print(df.head())


# --------------------------------------------------
# 3. DATA PREPROCESSING
# --------------------------------------------------

print("\n===== DATA PREPROCESSING =====")

# Create a copy
data = df.copy()

# Add a missing value for demonstration
data.loc[0, "sepal length (cm)"] = np.nan

print("\nMissing values before handling:")
print(data.isnull().sum())

# Fill missing numerical value with mean
data["sepal length (cm)"] = data["sepal length (cm)"].fillna(
    data["sepal length (cm)"].mean()
)

# Check missing values again
print("\nMissing values after handling:")
print(data.isnull().sum())

# Check duplicate rows
print("\nDuplicate rows:", data.duplicated().sum())

# Remove duplicate rows
data = data.drop_duplicates()

print("\nData preprocessing completed.")


# --------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# --------------------------------------------------

print("\n===== EDA =====")

print("\nDataset Shape:")
print(data.shape)

print("\nDataset Information:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

print("\nCorrelation Matrix:")
print(data.corr(numeric_only=True))


# --------------------------------------------------
# 5. DATA VISUALIZATION
# --------------------------------------------------

print("\n===== DATA VISUALIZATION =====")

# Histogram
plt.figure(figsize=(7, 5))
plt.hist(data["sepal length (cm)"], bins=10)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.title("Distribution of Sepal Length")
plt.show()


# Scatter Plot
plt.figure(figsize=(7, 5))
plt.scatter(
    data["sepal length (cm)"],
    data["petal length (cm)"]
)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Sepal Length vs Petal Length")
plt.show()


# Box Plot
plt.figure(figsize=(7, 5))
plt.boxplot(data["sepal width (cm)"])
plt.ylabel("Sepal Width (cm)")
plt.title("Box Plot of Sepal Width")
plt.show()


print("\nLab 3 completed successfully.")
