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
st.title("Data Science Jobs Salary Analysis")

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
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Salary by Experience
st.subheader("Salary Distribution by Experience Level")
fig1, ax1 = plt.subplots()
sns.boxplot(x='experience_level', y='salary_in_usd', data=df, ax=ax1)
st.pyplot(fig1)

# Top Job Titles
st.subheader("Average Salary for Top 10 Job Titles")
top_titles = df['job_title'].value_counts().head(10).index
avg_salary_by_title = df[df['job_title'].isin(top_titles)].groupby('job_title')['salary_in_usd'].mean().sort_values()
fig2, ax2 = plt.subplots()
avg_salary_by_title.plot(kind='barh', ax=ax2)
st.pyplot(fig2)

# Employment Type
st.subheader("Salary by Employment Type")
fig3, ax3 = plt.subplots()
sns.boxplot(x='employment_type', y='salary_in_usd', data=df, ax=ax3)
st.pyplot(fig3)

# Company Size
st.subheader("Salary by Company Size")
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
st.subheader("Model Performance")
st.write("**Linear Regression**")
st.write(f"R² Score: {lr_r2:.3f}")
st.write(f"RMSE: {lr_rmse:.2f}")

st.write("**Random Forest Regressor**")
st.write(f"R² Score: {rf_r2:.3f}")
st.write(f"RMSE: {rf_rmse:.2f}")

# Feature Importance
importances = rf.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

st.subheader("Feature Importance (Random Forest)")
fig5, ax5 = plt.subplots()
sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax5)
st.pyplot(fig5)




#def load_data(data_path:str):
#    return df

#df = load_data("data/salaries.csv")
#print(df.columns)

st.title("Data Dashboard")
st.markdown("## Sample Data")
st.write(df.head())

#st.write("Python")
#st.write(["Python", "JS", "Java"])
#st.divider()
#st.dataframe(df.head())

#st.markdown("##### Machine Learning Engineer")
def filter_job_title(df, column_value:str):
    custom_filter = df["job_title"] == column_value
    filtered_df = df[custom_filter]
    return filtered_df

#selected_column_value = "Machine Learning Engineer"
selected_column_value = st.text_input("Job Title","Machine Learning Engineer")
if selected_column_value:
    filtered_df = filter_job_title(df, selected_column_value)
    st.dataframe(filtered_df)

#SideBar
    #st.title("Average Salary")
#with st.sidebar:
   # st.subheader("Slider Selector:")
    #salary_range = st.slider("Select average salary range:", min_value= df["salary_in_usd"].min(), max_value=df["salary_in_usd"].max(), value=(0, 100000), step=10000)

    #st.write("Selected range:", salary_range)
#if "salary_in_usd" in df.columns:
    #salary_filtered_df = df[
        #(df["salary_in_usd"] >= salary_range[0]) & 
        #(df["salary_in_usd"] <= salary_range[1])
    #]
    #st.markdown("### Filtered Average Salary")
    #st.dataframe(salary_filtered_df)
#else:
    #st.error("Column 'salary_in_usd' not found in the dataset.")



#option = st.selectbox()


# Clean up
df['experience_level'] = df['experience_level'].map({
    'EN': 'Entry-level', 'MI': 'Mid-level', 'SE': 'Senior-level', 'EX': 'Executive-level'
})
df['employment_type'] = df['employment_type'].map({
    'PT': 'Part-time', 'FT': 'Full-time', 'CT': 'Contract', 'FL': 'Freelance'
})
df['company_size'] = df['company_size'].map({
    'S': 'Small', 'M': 'Medium', 'L': 'Large'
})

# Tabs
tab1, tab2, tab3 = st.tabs(["Dataset & Info", "Visual Analysis", "ML Models"])

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Basic Info")
    st.write(f"Total rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write("### Column Info")
    st.write(df.dtypes)

    st.subheader("Descriptive Statistics")
    st.write(df.describe())

with tab2:
    col1, col2, col3 = st.tabs(["Salary Distribution", "Job Title Salary", "Average Salary"])

    with col1:
        st.subheader("Salary Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['salary_in_usd'], kde=True, ax=ax, color='skyblue')
        ax.set_xlabel("Salary (USD)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    with col2:
        st.subheader("Average Salary by Top 10 Job Titles")
        top_titles = df['job_title'].value_counts().head(10).index
        avg_salary_by_title = df[df['job_title'].isin(top_titles)] \
                                .groupby('job_title')['salary_in_usd'].mean().sort_values()
        fig2, ax2 = plt.subplots()
        avg_salary_by_title.plot(kind='barh', ax=ax2)
        st.pyplot(fig2)

    with col3:
        st.subheader("Average Salary by Country")
        avg_salary_country = df.groupby('company_location')['salary_in_usd'].mean().sort_values(ascending=False)
        st.bar_chart(avg_salary_country)

with tab3:
    col1, col2, col3 = st.tabs(["Salary Prediction", "Remote Work vs Salary", "Job Filter"])
    with col1:
        st.subheader("ML Model: Salary Prediction")

        df_encoded = df.copy()

        for col in ['experience_level', 'employment_type', 'company_size',
                    'company_location', 'employee_residence', 'job_title']:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])

        # Drop rows with NaNs in relevant columns
        df_encoded = df_encoded.dropna(subset=['experience_level', 'employment_type', 'company_size', 'remote_ratio', 'salary_in_usd'])

        X = df_encoded[['experience_level', 'employment_type', 'company_size', 'remote_ratio']]
        y = df_encoded['salary_in_usd']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)

        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        from sklearn.metrics import mean_squared_error, r2_score

        st.markdown("#### Linear Regression Results")
        st.write(f"R² Score: {r2_score(y_test, y_pred_lr):.3f}")
        st.write(f"RMSE: {mean_squared_error(y_test, y_pred_lr, squared=False):.2f}")

        st.markdown("#### Random Forest Results")
        st.write(f"R² Score: {r2_score(y_test, y_pred_rf):.3f}")
        st.write(f"RMSE: {mean_squared_error(y_test, y_pred_rf, squared=False):.2f}")

        st.subheader("🔍 Feature Importance (Random Forest)")
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values(by='Importance', ascending=False)


        fig5, ax5 = plt.subplots()
        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax5)
        st.pyplot(fig5)
    with col2:
        st.subheader("Remote Work vs Salary")
        remote_map = {0: 'No Remote', 50: 'Hybrid', 100: 'Fully Remote'}
        df['remote_status'] = df['remote_ratio'].map(remote_map)

        fig, ax = plt.subplots()
        sns.boxplot(x='remote_status', y='salary_in_usd', data=df, palette='coolwarm', ax=ax)
        st.pyplot(fig)
    with col3:
        job_title = df['job_title'].dropna().unique().tolist()
        def filter_job_title(df, column_value:str):
            custom_filter = df["job_title"] == column_value
            filtered_df = df[custom_filter]
            return filtered_df
        selected_column_value = st.selectbox(
            "Job Title",
            job_title,
            index=None,
            placeholder="Select a saved title or enter a new one",
        )
        if selected_column_value:
            salary_filtered_df = filter_job_title(df, selected_column_value)
            st.dataframe(salary_filtered_df)
    
