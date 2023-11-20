import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load your dataset
df = pd.read_csv("ncbirths.csv")

# Title and description
st.title("Linear Regression Analysis with Streamlit")
st.write("This app explores a dataset and performs linear regression.")

# Sidebar for data exploration
st.sidebar.header("Data Exploration")

# Filter data by white mothers and male babies
filtered_df = df[(df["whitemom"] == "white") & (df["gender"] == "male")]

# Scatter plot: Weight vs. Mage
st.subheader("Weight vs. Mage")
st.pyplot(plt.scatter(filtered_df['mage'], filtered_df['weight'], alpha=0.5))
st.write("Scatter plot showing the relationship between Weight and Mage.")

# Scatter plot: Weight vs. Habit (Smoking Status)
st.subheader("Weight vs. Habit (Smoking Status)")
st.pyplot(plt.scatter(filtered_df['habit'], filtered_df['weight'], alpha=0.5))
st.write("Scatter plot showing the relationship between Weight and Habit (Smoking Status).")

# Scatter plot: Weight vs. Weeks
st.subheader("Weight vs. Weeks")
st.pyplot(plt.scatter(filtered_df['weeks'], filtered_df['weight'], alpha=0.5))
st.write("Scatter plot showing the relationship between Weight and Weeks.")

# Scatter plot: Weight vs. Visits
st.subheader("Weight vs. Visits")
st.pyplot(plt.scatter(filtered_df['visits'], filtered_df['weight'], alpha=0.5))
st.write("Scatter plot showing the relationship between Weight and Visits.")

# Scatter plot: Weight vs. Gained
st.subheader("Weight vs. Gained")
st.pyplot(plt.scatter(filtered_df['gained'], filtered_df['weight'], alpha=0.5))
st.write("Scatter plot showing the relationship between Weight and Gained.")

# Data preprocessing
st.sidebar.header("Data Preprocessing")
st.write("Now, let's preprocess the data.")

# Drop unnecessary columns
filtered_df.drop(columns=["fage", "marital", "mature", "premie", "lowbirthweight"], inplace=True)

# Histograms
st.subheader("Histograms")
numeric_columns = filtered_df.select_dtypes(include=["number"]).columns

# Set the overall figsize for the grid of histograms
fig, axes = plt.subplots(nrows=len(numeric_columns), ncols=1, figsize=(8, 4 * len(numeric_columns)))

# Iterate through numeric columns and create histograms
for i, column in enumerate(numeric_columns):
    ax = axes[i]
    sns.histplot(data=filtered_df, x=column, bins=20, kde=True, ax=ax)
    ax.set_title(f"Histogram of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

# Adjust spacing between subplots
plt.tight_layout()
st.pyplot(plt)

# Correlation matrix
st.subheader("Correlation Matrix")
correlation_matrix = filtered_df.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix")
st.pyplot(plt)

# Linear regression analysis with statsmodels
st.sidebar.header("Linear Regression Analysis")
st.write("Now, let's perform linear regression using statsmodels.")

# One-hot encoding for habit
filtered_df_encoded = pd.get_dummies(filtered_df, columns=["habit"], drop_first=True)

# Define independent and dependent variables
X = filtered_df_encoded[["mage", "weeks", "visits", "habit_smoker"]]
y = filtered_df_encoded["weight"]

# Add a constant term (intercept) to the independent variables
X = sm.add_constant(X)

# Create a linear regression model with statsmodels
model = sm.OLS(y, X)

# Fit the model
results = model.fit()

# Get the summary of the regression results
st.subheader("Regression Results")
st.text(results.summary())




