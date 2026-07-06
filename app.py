import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import streamlit.components.v1 as components
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Page Config
st.set_page_config(
    page_title='Customer Churn Prediction & Explainability Dashboard',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Load XGBoost Model
@st.cache_resource
def load_model():
    with open('models/xgb_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Load cleaned dataset to fit Encoders and Scalers so mapping is 100% matched to training
@st.cache_data
def load_reference_data():
    df = pd.read_csv('data/processed/churn_cleaned.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Preprocess
    ref_df = df.copy()
    ref_df['TotalCharges'].fillna(ref_df['TotalCharges'].median(), inplace=True)
    
    # 1. Encoders
    le_dict = {}
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                        'InternetService', 'OnlineSecurity', 'Contract', 'PaymentMethod']
    for col in categorical_cols:
        le = LabelEncoder()
        ref_df[col] = le.fit_transform(ref_df[col].astype(str))
        le_dict[col] = le
        
    # 2. Features
    ref_df['TenureGroup'] = pd.cut(ref_df['tenure'], bins=[-1, 12, 24, 48, 72], labels=[0, 1, 2, 3]).astype(int)
    # Map qcut statically to avoid dynamic sample counts
    ref_df['MonthlyChargesGroup'] = pd.cut(ref_df['MonthlyCharges'], bins=[-np.inf, 35.5, 70.35, 89.85, np.inf], labels=[0, 1, 2, 3]).astype(int)
    ref_df['TotalChargesPerMonth'] = ref_df['TotalCharges'] / (ref_df['tenure'] + 1)
    
    # Keep only target model features + Churn label
    model_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
                      'InternetService', 'OnlineSecurity', 'Contract', 'PaymentMethod', 
                      'MonthlyCharges', 'TotalCharges', 'TenureGroup', 'MonthlyChargesGroup', 'TotalChargesPerMonth']
    
    # 3. Scaler
    scaler = StandardScaler()
    scaler.fit(ref_df[model_features])
    
    return ref_df, model_features, le_dict, scaler

ref_df, model_features, le_dict, scaler = load_reference_data()

# ----------------- UI HEADERS -----------------
st.write(
    """
    <div style="background-color:#1e293b;padding:20px;border-radius:10px;margin-bottom:25px;">
        <h1 style="color:#f8fafc;margin:0;font-family:'Segoe UI',Roboto,Helvetica;font-weight:700;">📊 Customer Churn Prediction & Retention Dashboard</h1>
        <p style="color:#94a3b8;margin:5px 0 0 0;font-size:16px;">Evaluate individual customer churn risk and analyze model predictions with SHAP explainability.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------- SIDEBAR INPUTS -----------------
st.sidebar.header('👤 Enter Customer Attributes')

gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
senior_citizen = st.sidebar.selectbox('Senior Citizen', ['No', 'Yes'])
partner = st.sidebar.selectbox('Partner (Married/Cohabitant)', ['No', 'Yes'])
dependents = st.sidebar.selectbox('Dependents (Children/Elders)', ['No', 'Yes'])
tenure = st.sidebar.slider('Tenure (Contract Months)', 0, 72, 12)
phone_service = st.sidebar.selectbox('Phone Service', ['No', 'Yes'])
internet_service = st.sidebar.selectbox('Internet Service Type', ['DSL', 'Fiber optic', 'No'])
online_security = st.sidebar.selectbox('Online Security Addon', ['No', 'Yes', 'No internet service'])
contract = st.sidebar.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
payment_method = st.sidebar.selectbox('Payment Method', [
    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
])
monthly_charges = st.sidebar.slider('Monthly Charges ($)', 0, 120, 50)
total_charges = st.sidebar.slider('Total Charges ($)', 0, 8500, 1000)

# ----------------- INPUT PREPROCESSING -----------------
# 1. Map Inputs to match Training
input_dict = {
    'gender': le_dict['gender'].transform([gender])[0],
    'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
    'Partner': le_dict['Partner'].transform([partner])[0],
    'Dependents': le_dict['Dependents'].transform([dependents])[0],
    'tenure': tenure,
    'PhoneService': le_dict['PhoneService'].transform([phone_service])[0],
    'InternetService': le_dict['InternetService'].transform([internet_service])[0],
    'OnlineSecurity': le_dict['OnlineSecurity'].transform([online_security])[0],
    'Contract': le_dict['Contract'].transform([contract])[0],
    'PaymentMethod': le_dict['PaymentMethod'].transform([payment_method])[0],
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

# Derived columns
input_dict['TenureGroup'] = 0 if tenure <= 12 else 1 if tenure <= 24 else 2 if tenure <= 48 else 3
input_dict['MonthlyChargesGroup'] = 0 if monthly_charges <= 35.5 else 1 if monthly_charges <= 70.35 else 2 if monthly_charges <= 89.85 else 3
input_dict['TotalChargesPerMonth'] = total_charges / (tenure + 1)

# Build DF matching the correct model columns order
input_df = pd.DataFrame([input_dict])[model_features]

# Standardize features matching model training prep
input_scaled = pd.DataFrame(scaler.transform(input_df), columns=model_features)

# Run Prediction
pred_proba = model.predict_proba(input_scaled)[0][1]

# Display Risk Indicators
if pred_proba > 0.6:
    risk_color = "#ef4444"
    churn_risk = "HIGH 🔴"
elif pred_proba > 0.3:
    risk_color = "#eab308"
    churn_risk = "MEDIUM 🟡"
else:
    risk_color = "#22c55e"
    churn_risk = "LOW 🟢"

# --- Metrics Dashboard Row ---
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f"""
        <div style="background-color:#0f172a;padding:20px;border-radius:10px;text-align:center;border-left:5px solid {risk_color};">
            <h5 style="color:#94a3b8;margin:0;">Risk Category</h5>
            <h2 style="color:{risk_color};margin:5px 0 0 0;font-weight:700;">{churn_risk}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f"""
        <div style="background-color:#0f172a;padding:20px;border-radius:10px;text-align:center;">
            <h5 style="color:#94a3b8;margin:0;">Churn Probability</h5>
            <h2 style="color:#f8fafc;margin:5px 0 0 0;font-weight:700;">{pred_proba:.1%}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        f"""
        <div style="background-color:#0f172a;padding:20px;border-radius:10px;text-align:center;">
            <h5 style="color:#94a3b8;margin:0;">Retention Confidence</h5>
            <h2 style="color:#f8fafc;margin:5px 0 0 0;font-weight:700;">{(1-pred_proba):.1%}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- SECTION LAYOUT: RECOMENDS & SHAP -----------------
main_col1, main_col2 = st.columns([1, 1.3])

with main_col1:
    st.subheader("💡 Strategic Retention Playbook")
    if pred_proba > 0.6:
        st.warning('⚠️ **CRITICAL CHURN RISK LEVEL** — High probability of customer loss.')
        st.write(
            """
            **Immediate Prescriptions:**
            - **Financial Offer**: Recommend converting the customer from a month-to-month plan to a **2-year contract** with a promotional **20% discount**.
            - **Infrastructure Upgrade**: Upgrade the customer to fiber optic internet service, if available, at standard DSL rates.
            - **Loyalty Perks**: Proactively bundle a complimentary **Online Security addon** for the next 12 months.
            - **VIP Support**: Assign a dedicated client relationship associate for check-ins.
            """
        )
    elif pred_proba > 0.3:
        st.info('ℹ️ **ELEVATED CHURN RISK LEVEL** — Proactive outreach recommended.')
        st.write(
            """
            **Recommended Actions:**
            - **Promotional Offer**: Recommend transitioning to a **1-year contract** with a 10% discount.
            - **Ad-on Promotion**: Offer a trial of the Online Security or Device Protection bundle.
            - **Feedback Call**: Initiate a customer service follow-up to address any minor complaints.
            """
        )
    else:
        st.success('✅ **STABLE CUSTOMER** — Churn risk is within acceptable parameters.')
        st.write(
            """
            **Recommended Maintenance:**
            - Keep customer on current plan; do not interrupt with aggressive promotional offers.
            - Continue to send standardized seasonal updates and newsletters.
            """
        )

with main_col2:
    st.subheader("🔍 SHAP Explanation (Why This Prediction?)")
    try:
        # Create Explainer & Calculate forces
        explainer = shap.TreeExplainer(model)
        # Calculate SHAP values
        shap_vals = explainer.shap_values(input_scaled)
        
        # Resolve array dimension discrepancy
        if isinstance(shap_vals, list):
            cust_val = shap_vals[1]
            base_val = explainer.expected_value[1]
        elif len(np.shape(shap_vals)) == 3:
            cust_val = shap_vals[0, :, 1]
            base_val = explainer.expected_value[1]
        elif len(np.shape(shap_vals)) == 2:
            cust_val = shap_vals[0]
            base_val = explainer.expected_value
        else:
            cust_val = shap_vals
            base_val = explainer.expected_value
            
        # Draw the SHAP force plot and convert to HTML
        fig = shap.force_plot(
            base_val, 
            cust_val, 
            input_df.iloc[0], 
            matplotlib=False, 
            link='logit'
        )
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            shap.save_html(tmp.name, fig)
            with open(tmp.name, 'r') as html_file:
                shap_html = html_file.read()
        components.html(shap_html, height=135, scrolling=True)
        st.caption("🔴 Red bars push risk higher. 🔵 Blue bars keep risk lower. Hover over features for details.")
    except Exception as e:
        st.error(f"Could not load SHAP visualization: {e}")

st.markdown("---")
st.caption('💡 Retention model achieves peak 87% AUC metric on validation hold-outs.')