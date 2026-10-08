import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Paths
RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

def safe_load_csv(filename):
    path = TABLES_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    else:
        st.warning(f"File not found: {path}. Please ensure the corresponding experiment has been run.")
        return None

def safe_load_image(filename):
    path = FIGURES_DIR / filename
    if path.exists():
        return Image.open(path)
    else:
        st.warning(f"Image not found: {path}. Please ensure the corresponding experiment has been run.")
        return None

def page_overview():
    st.header("1. Overview")
    st.markdown("""
    **Project Objective:** Predict battery health from cycle data and estimate when a cell reaches end of life using NASA Li-ion aging data.
    
    **Batteries Used:**
    - B0005
    - B0006
    - B0007
    - B0018
    
    **Primary Features:**
    - `discharge_cycle_index`
    - `capacity` (from previous cycles)
    
    **Primary Target:**
    - Next-cycle discharge capacity.
    
    **Primary Split:**
    - Leave-One-Battery-Out (LOBO).
    
    **Leakage Demo Note:**
    Random-by-cycle splitting is deliberately treated as wrong in this project. Neighboring cycles of one cell are nearly identical. Mixing cycles from the same battery across train and test sets leads to severe data leakage and optimistically biased results. LOBO is the correct way to estimate cross-battery generalization.
    """)

def page_capacity_curves():
    st.header("2. Capacity Curves")
    img = safe_load_image("capacity_vs_cycle.png")
    if img:
        st.image(img, caption="Capacity vs. Cycle Index for all batteries", use_container_width=True)

def page_single_models():
    st.header("3. Single Models")
    df = safe_load_csv("single_model_lobo_results.csv")
    if df is not None:
        st.subheader("Full Results Table")
        st.dataframe(df)
        
        batteries = df['test_battery'].unique()
        selected_battery = st.selectbox("Select Held-out Battery", batteries, key="sm_battery")
        
        subset = df[df['test_battery'] == selected_battery]
        st.subheader(f"MAE by Model for {selected_battery}")
        st.bar_chart(data=subset, x='model', y='mae')
        
        st.subheader(f"RMSE by Model for {selected_battery}")
        st.bar_chart(data=subset, x='model', y='rmse')

def page_ensembles():
    st.header("4. Ensembles")
    df = safe_load_csv("ensemble_lobo_results.csv")
    if df is not None:
        st.subheader("Full Results Table")
        st.dataframe(df)
        
        batteries = df['test_battery'].unique()
        selected_battery = st.selectbox("Select Held-out Battery", batteries, key="ens_battery")
        
        subset = df[df['test_battery'] == selected_battery]
        st.subheader(f"MAE comparison for {selected_battery}")
        st.bar_chart(data=subset, x='model', y='mae')
        
        st.subheader(f"RMSE comparison for {selected_battery}")
        st.bar_chart(data=subset, x='model', y='rmse')
        
        st.subheader(f"Residual Correlation Heatmap for {selected_battery}")
        img = safe_load_image(f"residual_corr_{selected_battery}.png")
        if img:
            st.image(img, caption=f"Residual Correlation - {selected_battery}", use_container_width=True)

def page_classification_costs():
    st.header("5. Classification & Costs")
    df = safe_load_csv("classification_lobo_results.csv")
    if df is not None:
        st.subheader("Full Metrics Table")
        st.dataframe(df)
        
        batteries = df['test_battery'].unique()
        selected_battery = st.selectbox("Select Held-out Battery", batteries, key="clf_battery")
        
        subset = df[df['test_battery'] == selected_battery]
        st.subheader(f"Cost Comparison for {selected_battery}")
        st.bar_chart(data=subset, x='model', y='weighted_cost_per_row')
        
        st.subheader("TP/TN/FP/FN Values")
        st.dataframe(subset[['model', 'TP', 'TN', 'FP', 'FN']])
    
    st.subheader("Calibration Curves")
    cal_df = safe_load_csv("calibration_bins.csv")
    if cal_df is not None:
        st.markdown("**Note:** B0007 has zero positive examples under the chosen threshold. Its zero recall/F1 is expected and correctly shown.")
        
        if df is not None:
            cal_subset = cal_df[cal_df['test_battery'] == selected_battery]
            fig, ax = plt.subplots()
            
            for model_name, group in cal_subset.groupby('model'):
                ax.plot(group['mean_predicted_probability'], group['observed_positive_rate'], marker='o', label=model_name)
            
            ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated (y=x)')
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Observed Positive Rate')
            ax.set_title(f'Calibration for {selected_battery}')
            ax.legend()
            st.pyplot(fig)

def page_error_analysis():
    st.header("6. Error Analysis")
    df_region = safe_load_csv("error_by_life_region.csv")
    df_largest = safe_load_csv("largest_errors.csv")
    
    if df_region is not None and df_largest is not None:
        batteries = df_region['test_battery'].unique()
        selected_battery = st.selectbox("Select Held-out Battery", batteries, key="err_battery")
        
        st.subheader(f"Metrics by Life Region for {selected_battery}")
        region_subset = df_region[df_region['test_battery'] == selected_battery]
        st.dataframe(region_subset)
        
        st.subheader(f"Largest Errors for {selected_battery}")
        largest_subset = df_largest[df_largest['test_battery'] == selected_battery]
        st.dataframe(largest_subset)
        
        st.subheader(f"Error over Cycle for {selected_battery}")
        img = safe_load_image(f"error_over_cycle_{selected_battery}.png")
        if img:
            st.image(img, caption=f"Error vs Cycle - {selected_battery}", use_container_width=True)

def page_synthetic_shift():
    st.header("7. Synthetic Shift")
    st.markdown("**Note:** This is a controlled synthetic experiment, not a real deployment shift.")
    
    df = safe_load_csv("synthetic_shift_results.csv")
    if df is not None:
        st.subheader("Results")
        st.dataframe(df)
        
        if 'condition' in df.columns and 'mae' in df.columns:
            st.bar_chart(data=df, x='condition', y='mae')
    
    st.subheader("Synthetic Capacity Curves")
    img = safe_load_image("synthetic_capacity_curves.png")
    if img:
        st.image(img, caption="Controlled Synthetic Battery Data", use_container_width=True)

def main():
    st.title("Battery Health Prediction Lab")
    
    st.sidebar.title("Navigation")
    sections = {
        "1. Overview": page_overview,
        "2. Capacity Curves": page_capacity_curves,
        "3. Single Models": page_single_models,
        "4. Ensembles": page_ensembles,
        "5. Classification & Costs": page_classification_costs,
        "6. Error Analysis": page_error_analysis,
        "7. Synthetic Shift": page_synthetic_shift
    }
    
    selection = st.sidebar.radio("Go to", list(sections.keys()))
    sections[selection]()

if __name__ == "__main__":
    main()
