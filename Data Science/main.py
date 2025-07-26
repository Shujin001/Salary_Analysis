#Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(data_path:str):
    df = pd.read_csv(data_path)
    return df

df = load_data("data/salaries.csv")
print(df.columns)

st.set_page_config("Salaries Analysis")

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
    st.title("Average Salary")
with st.sidebar:
    st.subheader("Slider Selector:")
    salary_range = st.slider("Select average salary range:", min_value= df["salary_in_usd"].min(), max_value=df["salary_in_usd"].max(), value=(0, 100000), step=10000)

    st.write("Selected range:", salary_range)
if "salary_in_usd" in df.columns:
    salary_filtered_df = df[
        (df["salary_in_usd"] >= salary_range[0]) & 
        (df["salary_in_usd"] <= salary_range[1])
    ]
    st.markdown("### Filtered Average Salary")
    st.dataframe(salary_filtered_df)
else:
    st.error("Column 'salary_in_usd' not found in the dataset.")

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

#option = st.selectbox()


tab1, tab2, tab3 = st.tabs(["tab1","tab2","tab3"])
with tab1:
    #st.dataframe(filtered_df)
    st.write("TAB1")
with tab2:
    st.write("TAB2")
with tab3:
    st.write("TAB3")
col1, col2, col3 = st.tabs(["col1","col2","col3"])
with col1:
    st.write("COL1")
with col2:
    st.write("COL2")
with col3:
    st.write("COL3")