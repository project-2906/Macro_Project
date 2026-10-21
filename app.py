import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Bank Churn Predictor", layout="wide")

# --- CUSTOM CSS (Professional Theme) ---
st.markdown("""
<style>
    /* Background */
    .stApp {
        background-color: #FFF5F6;
        background-image: url("https://www.transparenttextures.com/patterns/floral-white.png");
    }
    
    /* Main Card Styling */
    div[data-testid="stForm"] {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(255, 182, 193, 0.4);
        border: 1px solid #FFC0CB;
    }
    
    /* Headings */
    h1, h2, h3, h4 {
        color: #D87093 !important;
        font-family: 'Georgia', serif;
        text-align: center;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #FFB6C1;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #FF69B4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR (MODEL METRICS & EDA CHART) ---
with st.sidebar:
    st.markdown("<h2 style='text-align: left;'>Model Information</h2>", unsafe_allow_html=True)
    st.write("**Algorithm:** XGBoost Classifier")
    st.write("**Accuracy:** 84.55%")
    st.write("**F1-Score:** 62.64%")
    st.write("**Dataset:** 10,000 Bank Customers")
    st.write("---")
    
    # --- INTERACTIVE EDA CHART ---
    st.markdown("<h3 style='text-align: left; font-size: 18px;'>Key Drivers of Churn</h3>", unsafe_allow_html=True)
    
    # Hardcoded feature importances from your XGBoost model
    features = ['Age', 'NumOfProducts', 'IsActiveMember', 'Balance', 'Geography_Germany', 'Gender', 'Tenure', 'EstimatedSalary', 'CreditScore', 'Geography_Spain']
    importances = [0.39, 0.25, 0.11, 0.08, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01]
    
    # Create a DataFrame for Plotly
    df_eda = pd.DataFrame({'Feature': features, 'Importance': importances})
    
    # Create interactive bar chart
    fig_eda = px.bar(
        df_eda, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        color='Importance',
        color_continuous_scale=['#FFB6C1', '#D87093'] # Floral theme colors
    )
    
    fig_eda.update_layout(
        height=300, 
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis={'categoryorder':'total ascending'},
        coloraxis_showscale=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_eda, use_container_width=True)
    st.write("---")
    st.markdown("<p style='font-size:12px; color:gray;'>This model was trained using SMOTE to balance the data and GridSearchCV for optimization.</p>", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1>Bank Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4>Predict if a customer will leave your bank</h4>", unsafe_allow_html=True)
st.write("---")

# --- LOAD MODEL ---
try:
    model = joblib.load('churn_xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
except:
    st.error("Error: Model or scaler file not found! Please ensure 'churn_xgb_model.pkl' and 'scaler.pkl' are in the same folder.")

# --- TABS (Single vs Batch) ---
tab1, tab2 = st.tabs(["Single Customer Prediction", "Batch Prediction (Upload File)"])

# ==========================================
# TAB 1: SINGLE CUSTOMER PREDICTION
# ==========================================
with tab1:
    with st.form("prediction_form"):
        st.markdown("<h4>Enter Customer Details:</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)
            balance = st.number_input("Account Balance ($)", min_value=0.0, value=75000.0)
            estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0)
            
        with col2:
            geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
            has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
            is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
            
        submitted = st.form_submit_button("Predict Churn Risk")

    if submitted:
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [1 if gender == "Male" else 0],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_products],
            'HasCrCard': [1 if has_card == "Yes" else 0],
            'IsActiveMember': [1 if is_active == "Yes" else 0],
            'EstimatedSalary': [estimated_salary],
            'Geography_Germany': [1 if geography == "Germany" else 0],
            'Geography_Spain': [1 if geography == "Spain" else 0]
        })
        
        num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
        input_data[num_cols] = scaler.transform(input_data[num_cols])
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        st.write("---")
        
        # Split into two columns for Summary and Gauge
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.markdown(f"### Customer Summary")
            st.info(f"**Profile:** {age}-year-old {gender} from {geography}.\n\n"
                    f"**Financials:** Balance of ${balance:,.2f} with {num_products} product(s).\n\n"
                    f"**Status:** {'Active' if is_active == 'Yes' else 'Inactive'} member with {tenure} years of tenure.")
            
            st.markdown(f"### Recommendation")
            if prediction == 1:
                st.error(f"**High Risk!** This customer is likely to churn.")
                if age > 50:
                    st.warning("Older customers have a high churn rate. Offer a dedicated senior account manager.")
                elif num_products >= 3:
                    st.warning("This customer has too many products. Schedule a call to simplify their banking experience.")
                elif is_active == "No":
                    st.warning("The customer is inactive. Send a re-engagement email with a small cash bonus.")
                else:
                    st.warning("Offer a short-term retention discount on their account fees.")
            else:
                st.success(f"**Safe!** This customer is likely to stay.")
                st.info("Maintain regular engagement and offer cross-selling opportunities.")

        with res_col2:
            # PLOTLY GAUGE CHART
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Probability", 'font': {'size': 24, 'color': '#D87093'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#D87093"},
                    'bar': {'color': "#D87093"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "lightgray",
                    'steps': [
                        {'range': [0, 40], 'color': "#E8F8F5"},   # Safe (Light Green)
                        {'range': [40, 70], 'color': "#FEF9E7"},  # Warning (Light Yellow)
                        {'range': [70, 100], 'color': "#FDEDEC"}  # High Risk (Light Red)
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: BATCH PREDICTION
# ==========================================
with tab2:
    st.markdown("<h4>Upload a CSV file to predict churn for multiple customers at once.</h4>", unsafe_allow_html=True)
    st.markdown("<p>The file must contain the following columns exactly: <b>CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary</b></p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(batch_data.head())
        
        if st.button("Run Batch Prediction"):
            try:
                df_processed = batch_data.copy()
                df_processed['Gender'] = df_processed['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
                df_processed['Geography_Germany'] = df_processed['Geography'].apply(lambda x: 1 if x == 'Germany' else 0)
                df_processed['Geography_Spain'] = df_processed['Geography'].apply(lambda x: 1 if x == 'Spain' else 0)
                df_processed = df_processed.drop('Geography', axis=1)
                
                feature_order = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain']
                df_features = df_processed[feature_order]
                
                num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
                df_features[num_cols] = scaler.transform(df_features[num_cols])
                
                predictions = model.predict(df_features)
                probabilities = model.predict_proba(df_features)[:, 1]
                
                batch_data['Churn_Prediction'] = predictions
                batch_data['Churn_Probability'] = probabilities
                
                st.success("Batch prediction completed successfully!")
                st.dataframe(batch_data)
                
                csv = batch_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name='churn_predictions.csv',
                    mime='text/csv',
                )
                
            except Exception as e:
                st.error(f"An error occurred: {e}. Please ensure the CSV format matches the instructions.")