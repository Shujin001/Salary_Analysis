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
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Data Science Salary Analysis", layout="wide")

# --- SESSION STATE FOR NAVIGATION ---
if "page" not in st.session_state:
    st.session_state.page = "home"  # default page

# --- TYPEWRITER FUNCTION --- 
def typewriter_text(text, placeholder, delay=0.03):
    typed = ""
    for char in text:
        typed += char
        placeholder.markdown(typed)
        time.sleep(delay)

# --- HOME PAGE ---
if st.session_state.page == "home":
    # --- PLACEHOLDERS FOR TYPEWRITER ---
    title_placeholder = st.empty()
    subtitle_placeholder = st.empty()
    markdown_placeholder = st.empty()
    name_placeholder = st.empty()
    text_placeholder = st.empty()

    # --- TYPEWRITER FUNCTION --- 
    def typewriter_text(text, placeholder, delay=0.03, font_size=70, italic=False):
        typed = ""
        style = "italic" if italic else "normal"

        for char in text:
            typed += char
            placeholder.markdown(
                f"<h1 style='text-align:center; font-style:{style}; font-size:{font_size}px;'>{typed}</h1>",
                unsafe_allow_html=True
            )
            time.sleep(delay)


    if "typed" not in st.session_state:
        st.session_state.typed = False
    
    if not st.session_state.typed:
    # --- TYPEWRITER TEXT ---
        typewriter_text("💼 Data Science Jobs Salary Analysis", title_placeholder, font_size=70)
        typewriter_text("Welcome to the Salary Insights Dashboard 📊", subtitle_placeholder, font_size=35)

        typewriter_text("Discover trends, analyze patterns, and explore salaries across different roles, experience levels, and company sizes in the data science industry.", markdown_placeholder, font_size=19, delay=0.01)
        typewriter_text("Made by Chirag Sharma", name_placeholder, font_size=20, delay=0.01, italic=True)
        typewriter_text("🚀 Click below to start your journey!", text_placeholder, font_size=18, delay=0.01)

        st.session_state.typed = True
    
    if st.session_state != "analysis":
        if st.button("👉 Start Analysis"):
            st.session_state.page = "analysis"
            st.rerun()
# --- ANALYSIS PAGE ---
elif st.session_state.page == "analysis":
    # Load dataset
    if st.sidebar.button("⬅️ Back to Home"):
        st.session_state.clear()
        st.session_state.page = "home"
        st.session_state.typed = False
        st.rerun()
    try:
        df = pd.read_csv("Analysis/data/salaries.csv")
    except FileNotFoundError:
        st.error("Error: 'salaries.csv' not found in 'data' directory.")
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

    # Sidebar filters
    min_salary = int(df['salary_in_usd'].min())
    max_salary = int(df['salary_in_usd'].max())
    salary_range = st.sidebar.slider("🔎 Filters", min_salary, max_salary, (min_salary, max_salary), step=1000)

    experience_options = sorted(df['experience_level'].unique().tolist())
    selected_experience = st.sidebar.multiselect("Experience Level", experience_options, default=experience_options)

    employment_options = sorted(df['employment_type'].unique().tolist())
    selected_employment = st.sidebar.multiselect("Employment Type", employment_options, default=employment_options)

    company_options = sorted(df['company_size'].unique().tolist())
    selected_company = st.sidebar.multiselect("Company Size", company_options, default=company_options)

    # Apply filters
    df_filtered = df[
        (df['salary_in_usd'] >= salary_range[0]) &
        (df['salary_in_usd'] <= salary_range[1]) &
        (df['experience_level'].isin(selected_experience)) &
        (df['employment_type'].isin(selected_employment)) &
        (df['company_size'].isin(selected_company))
    ].copy()

    if df_filtered.empty:
        st.warning("No data available for the selected filters.")

    sns.set_theme(style="whitegrid")

    # Tabs for sections
    tab1, tab2, tab3 = st.tabs(["🗒️ Dataset & Info", "📈 Visual Analysis", "🤖 ML Models"])

    with tab1:
        st.header("📋 Dataset Overview")
        st.markdown(f"**Total Rows:** {df_filtered.shape[0]}    **Columns:** {df_filtered.shape[1]}")
        st.subheader("🗂️ Data Preview")
        st.dataframe(df_filtered.head())

        with st.expander("ℹ️ Column Info"):
            st.write(df_filtered.dtypes)
        with st.expander("📊 Descriptive Statistics"):
            st.write(df_filtered.describe())

    with tab2:
        st.header("📊 Salary Visualizations")

        # Salary Distribution
        st.subheader("💰 Salary Distribution")
        fig_dist, ax_dist = plt.subplots(figsize=(6,4))
        sns.histplot(df_filtered['salary_in_usd'], kde=True, color='skyblue', ax=ax_dist)
        st.pyplot(fig_dist)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💼 Salary by Experience Level")
            fig_exp, ax_exp = plt.subplots(figsize=(6,4))
            sns.boxplot(x='experience_level', y='salary_in_usd', data=df_filtered, palette='pastel', ax=ax_exp)
            st.pyplot(fig_exp)

            st.subheader("🛠️ Salary by Employment Type")
            fig_emp, ax_emp = plt.subplots(figsize=(6,4))
            sns.boxplot(x='employment_type', y='salary_in_usd', data=df_filtered, palette='Set2', ax=ax_emp)
            st.pyplot(fig_emp)

        with col2:
            st.subheader("🏢 Salary by Company Size")
            fig_size, ax_size = plt.subplots(figsize=(6,4))
            sns.barplot(x='company_size', y='salary_in_usd', data=df_filtered, palette='coolwarm', ax=ax_size)
            st.pyplot(fig_size)

            st.subheader("🚀 Top 10 Job Titles (Avg Salary)")
            top_titles = df_filtered['job_title'].value_counts().head(10).index
            avg_salary_by_title = df_filtered[df_filtered['job_title'].isin(top_titles)] \
                                    .groupby('job_title')['salary_in_usd'].mean().sort_values()
            fig_title, ax_title = plt.subplots(figsize=(6,4))
            avg_salary_by_title.plot(kind='barh', color='seagreen', ax=ax_title)
            st.pyplot(fig_title)

        # 👇 wrap model training in a cached function
    @st.cache_resource
    def train_models(X_train, y_train):
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train, y_train)

        return lr, rf

    with tab3:
        st.header("🤖 Model Performance")
        if st.button("⚡ Run Models"):
            df_model = df_filtered.copy()
            le = LabelEncoder()
            for col in ['experience_level', 'employment_type', 'company_size',
                        'company_location', 'employee_residence', 'job_title']:
                df_model[col] = le.fit_transform(df_model[col])
            df_model = df_model.dropna(subset=['experience_level','employment_type',
                                            'company_size','remote_ratio','salary_in_usd'])

            X = df_model[['experience_level', 'employment_type', 'company_size', 'remote_ratio']]
            y = df_model['salary_in_usd']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 🚀 models are only trained once unless inputs change
            lr, rf = train_models(X_train, y_train)

            y_pred_lr = lr.predict(X_test)
            y_pred_rf = rf.predict(X_test)

            lr_r2 = r2_score(y_test, y_pred_lr)
            lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
            rf_r2 = r2_score(y_test, y_pred_rf)
            rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Linear Regression")
                st.write(f"R² Score: **{lr_r2:.3f}**")
                st.write(f"RMSE: **{lr_rmse:.2f}**")
            with col2:
                st.subheader("Random Forest")
                st.write(f"R² Score: **{rf_r2:.3f}**")
                st.write(f"RMSE: **{rf_rmse:.2f}**")

            st.subheader("🌟 Feature Importance (Random Forest)")
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            fig_imp, ax_imp = plt.subplots(figsize=(6,4))
            sns.barplot(x='Importance', y='Feature', data=importance_df, palette='magma', ax=ax_imp)
            st.pyplot(fig_imp)

