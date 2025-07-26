#Importing Libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Page config
st.set_page_config(page_title="Data Science Salary Analysis", layout="wide")
st.title("📊 Data Science Jobs Salary Analysis")

# Load dataset automatically
try:
    df = pd.read_csv("data/salaries.csv")
except FileNotFoundError:
    st.error("'salaries.csv' not found in current directory.")
    st.stop()

# Map categorical codes to readable values
df['experience_level'] = df['experience_level'].map({
    'EN': 'Entry-level', 'MI': 'Mid-level', 'SE': 'Senior-level', 'EX': 'Executive-level'
})
df['employment_type'] = df['employment_type'].map({
    'PT': 'Part-time', 'FT': 'Full-time', 'CT': 'Contract', 'FL': 'Freelance'
})
df['company_size'] = df['company_size'].map({
    'S': 'Small', 'M': 'Medium', 'L': 'Large'
})

# Preview
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

# Salary by Experience
st.subheader("📈 Salary Distribution by Experience Level")
fig1, ax1 = plt.subplots()
sns.boxplot(x='experience_level', y='salary_in_usd', data=df, ax=ax1)
st.pyplot(fig1)

# Top Job Titles
st.subheader("📊 Average Salary for Top 10 Job Titles")
top_titles = df['job_title'].value_counts().head(10).index
avg_salary_by_title = df[df['job_title'].isin(top_titles)].groupby('job_title')['salary_in_usd'].mean().sort_values()
fig2, ax2 = plt.subplots()
avg_salary_by_title.plot(kind='barh', ax=ax2)
st.pyplot(fig2)

# Employment Type
st.subheader("💼 Salary by Employment Type")
fig3, ax3 = plt.subplots()
sns.boxplot(x='employment_type', y='salary_in_usd', data=df, ax=ax3)
st.pyplot(fig3)

# Company Size
st.subheader("🏢 Salary by Company Size")
fig4, ax4 = plt.subplots()
sns.barplot(x='company_size', y='salary_in_usd', data=df, ax=ax4)
st.pyplot(fig4)

# Model section
df_encoded = df.copy()
le = LabelEncoder()
for col in ['experience_level', 'employment_type', 'company_size', 'company_location', 'employee_residence', 'job_title']:
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded[['experience_level', 'employment_type', 'company_size', 'remote_ratio']]
y = df_encoded['salary_in_usd']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
lr_r2 = r2_score(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))

# Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# Model Output
st.subheader("📊 Model Performance")
st.write("**Linear Regression**")
st.write(f"R² Score: {lr_r2:.3f}")
st.write(f"RMSE: {lr_rmse:.2f}")

st.write("**Random Forest Regressor**")
st.write(f"R² Score: {rf_r2:.3f}")
st.write(f"RMSE: {rf_rmse:.2f}")

# Feature Importance
importances = rf.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

st.subheader("🔍 Feature Importance (Random Forest)")
fig5, ax5 = plt.subplots()
sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax5)
st.pyplot(fig5)



