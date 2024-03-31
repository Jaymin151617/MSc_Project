# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import requests
import lime
import lime.lime_tabular

# Code for the information related to the app
linkedin = "https://www.linkedin.com/in/jaymin-mistry-765902212"
st.header('Loan Approval Predictor', divider='red')
st.caption('This app was built as a part of my Final Year MSc Project. It aims to predict whether the user will get a loan or not based on the information he/she provides.\
           To use the app just fill the boxes below and generate predictions.')
st.caption(f'Connect with me on LinkedIn [here]({linkedin}).')

# Enter features values
col1, col2 = st.columns(2)
with col1:
    income = st.number_input("Annual Income", min_value=200000, max_value=9900000,step=1000, format="%d")
    graduate = st.checkbox("Graduate")
    loan_term = st.number_input("Loan Term (years)", min_value=2, max_value=20,step=1, format="%d")
    residential_assets_value = st.number_input("Residential Assets Value", min_value=-100000, max_value=29100000,step=10000, format="%d")
with col2:
    dependents = st.selectbox("No. of Dependents", list(range(6)))
    self_employed = st.checkbox("Self Employed")
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900,step=10, format="%d")
    commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, max_value=19400000, step=10000, format="%d")
loan_amount = st.number_input("**Enter Loan Amount**", min_value=300000, max_value=39500000 ,step=10000)
debt_income_ratio = loan_amount / income

# Store features values
values = {
    'debt_income_ratio':debt_income_ratio,
    'no_of_dependents': dependents,
    'education': int(graduate),
    'self_employed': int(self_employed),
    'loan_term': loan_term,
    'cibil_score': cibil_score,
    'residential_assets_value': np.cbrt(residential_assets_value),
    'commercial_assets_value': np.cbrt(commercial_assets_value)
}

# Convert dictionary to DataFrame
df = pd.DataFrame([values])

@st.cache_resource
def load_model(url):
    # Download the model file
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    
    # Load the model from the downloaded file
    model = pickle.loads(response.content)
    return model

@st.cache_resource
def load_X_train():
    # Load X_train data from a CSV file (replace "X_train.csv" with your file path)
    X_train = pd.read_csv('https://raw.githubusercontent.com/Jaymin151617/MSc_Project/main/X_train.csv')
    return X_train

def generate_predictions():
    # Load the Min-Max scaler
    scaler = load_model('https://raw.githubusercontent.com/Jaymin151617/MSc_Project/main/scaler.pkl')

    # Transform the DataFrame using the loaded scaler
    pred = scaler.transform(df)
    pred_df = pd.DataFrame(pred, columns=df.columns)

    # Load the XGBoost model
    xgb_model = load_model('https://raw.githubusercontent.com/Jaymin151617/MSc_Project/main/xgb_model.pkl')

    # Use the XGBoost model to predict the probability
    probability = xgb_model.predict_proba(pred_df)[:,1] * 100

    # Load X_train data and cache it
    X_train = load_X_train()

    # Initialize the LimeTabularExplainer
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['Rejected', 'Good'])

    # Generate explanation
    explanation = explainer.explain_instance(pred_df.iloc[0].values, xgb_model.predict_proba)

    # Get feature importance and names
    feature_importance = explanation.as_list()
    features = [x[0] for x in feature_importance]
    importance = [x[1] for x in feature_importance]

    # Sort features by importance
    sorted_indices = np.argsort(importance)[::-1]
    features = [features[i] for i in sorted_indices]
    importance = [importance[i] for i in sorted_indices]

    # Plot bar chart
    plt.figure(figsize=(8, 6))
    bars = plt.barh(features, importance, color=['blue' if imp >= 0 else 'orange' for imp in importance])
    plt.ylabel('Feature')
    plt.title('Feature Contribution')
    plt.gca().invert_yaxis()  # Invert y axis to have the most important features at the top
    plt.xticks([])  # Remove ticks on x axis

    # Add legend
    legend_labels = ['Increases %', 'Decreases %']
    legend_patches = [plt.Rectangle((0,0),1,1, color='blue'), plt.Rectangle((0,0),1,1, color='orange')]
    plt.legend(legend_patches, legend_labels)

    # Display the predicted probability as styled text
    st.markdown(f'<p style="font-size:20px; font-weight:bold;">Chances of loan approval for current loan amount: {probability[0]:.4f}%</p>', unsafe_allow_html=True)
    st.pyplot(plt)

# Create a Streamlit button
if st.button("Generate Predictions"):
    # Call the function to generate predictions when the button is clicked
    generate_predictions()