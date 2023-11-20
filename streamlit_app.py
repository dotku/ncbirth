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
st.pyplot(fig)


# Scatter plot: Weight vs. Mage
st.subheader("Weight vs. Mage")
scatter_fig_mage = plt.figure()
plt.scatter(filtered_df['mage'], filtered_df['weight'], alpha=0.5)
plt.xlabel('Mage')
plt.ylabel('Weight')
plt.title('Weight vs. Mage')
st.pyplot(scatter_fig_mage)

# Scatter plot: Weight vs. Habit (Smoking Status)
st.subheader("Weight vs. Habit (Smoking Status)")
scatter_fig_habit = plt.figure()
plt.scatter(filtered_df['habit'], filtered_df['weight'], alpha=0.5)
plt.xlabel('Habit (Smoking Status)')
plt.ylabel('Weight')
plt.title('Weight vs. Habit (Smoking Status)')
st.pyplot(scatter_fig_habit)

# Scatter plot: Weight vs. Weeks
st.subheader("Weight vs. Weeks")
scatter_fig_weeks = plt.figure()
plt.scatter(filtered_df['weeks'], filtered_df['weight'], alpha=0.5)
plt.xlabel('Weeks')
plt.ylabel('Weight')
plt.title('Weight vs. Weeks')
st.pyplot(scatter_fig_weeks)

# Scatter plot: Weight vs. Visits
st.subheader("Weight vs. Visits")
scatter_fig_visits = plt.figure()
plt.scatter(filtered_df['visits'], filtered_df['weight'], alpha=0.5)
plt.xlabel('Visits')
plt.ylabel('Weight')
plt.title('Weight vs. Visits')
st.pyplot(scatter_fig_visits)

# Scatter plot: Weight vs. Gained
st.subheader("Weight vs. Gained")
scatter_fig_gained = plt.figure()
plt.scatter(filtered_df['gained'], filtered_df['weight'], alpha=0.5)
plt.xlabel('Gained')
plt.ylabel('Weight')
plt.title('Weight vs. Gained')
st.pyplot(scatter_fig_gained)

# Data preprocessing
st.sidebar.header("Data Preprocessing")
st.write("Now, let's preprocess the data.")

# Linear regression analysis with statsmodels
st.sidebar.header("Linear Regression Analysis")
st.write("Now, let's perform linear regression using statsmodels.")

# Create a copy of the filtered DataFrame to avoid SettingWithCopyWarning
filtered_df = filtered_df.copy()

# Convert columns to numeric
filtered_df['weight'] = pd.to_numeric(filtered_df['weight'], errors='coerce')
filtered_df['mage'] = pd.to_numeric(filtered_df['mage'], errors='coerce')
filtered_df['weeks'] = pd.to_numeric(filtered_df['weeks'], errors='coerce')
filtered_df['visits'] = pd.to_numeric(filtered_df['visits'], errors='coerce')
filtered_df['weeks'].fillna(df['weeks'].mean(), inplace=True)
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
st.text(results.summary().as_text())






