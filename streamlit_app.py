
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

#Page configuration
st.set_page_config(
    page_title="Income Level Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

#css styles 
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-success {
        background-color: #d4edda;
        border: 2px solid #28a745;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .prediction-warning {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

#Load model and preprocessing objects
@st.cache_resource
def load_model():
    """Load trained model and preprocessing objects"""
    try:
        model = joblib.load('income_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return model, scaler, feature_names, metadata
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.stop()

model, scaler, feature_names, metadata = load_model()

#Header
st.markdown('<div class="main-header"> Income Level Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict whether an individual earns more than $50,000 annually</div>', unsafe_allow_html=True)

# Display model performance
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Model", metadata['model_name'].split()[0])
with col2:
    st.metric("F1-Score", f"{metadata['f1_score']:.3f}")
with col3:
    st.metric("Accuracy", f"{metadata['accuracy']:.3f}")
with col4:
    st.metric("Precision", f"{metadata['precision']:.3f}")
with col5:
    st.metric("Recall", f"{metadata['recall']:.3f}")

st.markdown("---")

#User inputs in sidebar
st.sidebar.header("Enter Individual Details")
st.sidebar.markdown("Fill in the information below to predict income level")

# Demographic Information
st.sidebar.subheader(" Demographic Information")
age = st.sidebar.slider("Age", 17, 90, 35, help="Age of the individual")
sex = st.sidebar.selectbox("Sex", ['Male', 'Female'])
race = st.sidebar.selectbox("Race", 
    ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])

# Education
st.sidebar.subheader(" Education")
education_num = st.sidebar.slider("Years of Education", 1, 16, 13, 
    help="Number of years of formal education (e.g., 13=Bachelor's degree)")

# Work Information
st.sidebar.subheader(" Work Information")
workclass = st.sidebar.selectbox("Work Class", 
    ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
     'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', 'Unknown'])

occupation = st.sidebar.selectbox("Occupation",
    ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 
     'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 
     'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 
     'Transport-moving', 'Priv-house-serv', 'Protective-serv', 
     'Armed-Forces', 'Unknown'])

hours_per_week = st.sidebar.slider("Hours Worked per Week", 1, 99, 40)

# Family Information
st.sidebar.subheader(" Family Information")
marital_status = st.sidebar.selectbox("Marital Status",
    ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 
     'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])

relationship = st.sidebar.selectbox("Relationship",
    ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 
     'Wife', 'Other-relative'])

# Financial Information
st.sidebar.subheader(" Financial Information")
capital_gain = st.sidebar.number_input("Capital Gain ($)", 0, 100000, 0, step=1000,
    help="Profit from sale of assets (stocks, property, etc.)")
capital_loss = st.sidebar.number_input("Capital Loss ($)", 0, 5000, 0, step=100,
    help="Loss from sale of assets")

# Geographic Information
st.sidebar.subheader(" Geographic Information")
native_country = st.sidebar.selectbox("Native Country",
    ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada',
     'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica',
     'Other'])

#Main predictin model

# Show input summary
st.subheader(" Input Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Personal Info**")
    st.write(f"• Age: {age}")
    st.write(f"• Sex: {sex}")
    st.write(f"• Race: {race}")
    st.write(f"• Native Country: {native_country}")

with col2:
    st.markdown("**Work Info**")
    st.write(f"• Work Class: {workclass}")
    st.write(f"• Occupation: {occupation}")
    st.write(f"• Hours/Week: {hours_per_week}")
    st.write(f"• Education: {education_num} years")

with col3:
    st.markdown("**Financial Info**")
    st.write(f"• Marital Status: {marital_status}")
    st.write(f"• Relationship: {relationship}")
    st.write(f"• Capital Gain: ${capital_gain:,}")
    st.write(f"• Capital Loss: ${capital_loss:,}")

st.markdown("---")

# Predict button
if st.button("Predict Income Level", type="primary", use_container_width=True):
    
    with st.spinner("Analyzing input data..."):
        
#Feature engineering
        has_capital_gain = 1 if capital_gain > 0 else 0
        has_capital_loss = 1 if capital_loss > 0 else 0
        is_married = 1 if 'Married' in marital_status else 0
        high_education = 1 if education_num >= 13 else 0
        
#Create input DataFrame 
        input_data = pd.DataFrame({
            'age': [age],
            'education_num': [education_num],
            'hours_per_week': [hours_per_week],
            'workclass': [workclass],
            'marital_status': [marital_status],
            'occupation': [occupation],
            'relationship': [relationship],
            'race': [race],
            'sex': [sex],
            'native_country_grouped': [native_country],
            'has_capital_gain': [has_capital_gain],
            'has_capital_loss': [has_capital_loss],
            'is_married': [is_married],
            'high_education': [high_education]
        })
        
#One hot encoding
        input_encoded = pd.get_dummies(input_data, 
            columns=['workclass', 'marital_status', 'occupation', 
                     'relationship', 'race', 'sex', 'native_country_grouped'],
            drop_first=True, dtype=int)
        
        # Align with training features
        for col in feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[feature_names]
        
#Scale numeric features
        numeric_cols = ['age', 'education_num', 'hours_per_week']
        input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])
        
#Make prediction
        prediction = model.predict(input_encoded)[0]
        prediction_proba = model.predict_proba(input_encoded)[0]
        
#Display results
    st.markdown("##  Prediction Results")
    
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-success">
            <h2 style="color: #28a745; margin: 0;"> Predicted Income: >$50,000</h2>
            <p style="font-size: 1.1rem; margin-top: 0.5rem;">
                This individual is predicted to earn <strong>more than $50,000</strong> annually.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-warning">
            <h2 style="color: #856404; margin: 0;"> Predicted Income: ≤$50,000</h2>
            <p style="font-size: 1.1rem; margin-top: 0.5rem;">
                This individual is predicted to earn <strong>$50,000 or less</strong> annually.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability breakdown
    st.markdown("### Confidence Breakdown")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Probability: ≤$50K",
            f"{prediction_proba[0]*100:.1f}%",
            delta=None
        )
        st.progress(prediction_proba[0])
    
    with col2:
        st.metric(
            "Probability: >$50K",
            f"{prediction_proba[1]*100:.1f}%",
            delta=None
        )
        st.progress(prediction_proba[1])
    
    # Model confidence interpretation
    confidence = max(prediction_proba[0], prediction_proba[1])
    st.markdown("###  Confidence Interpretation")
    
    if confidence >= 0.9:
        st.success(" **Very High Confidence** - The model is very certain about this prediction.")
    elif confidence >= 0.75:
        st.info(" **High Confidence** - The model is confident about this prediction.")
    elif confidence >= 0.6:
        st.warning(" **Moderate Confidence** - The model has reasonable confidence in this prediction.")
    else:
        st.error(" **Low Confidence** - The model is uncertain about this prediction. Results may vary.")

#Example scenarios
st.markdown("---")
with st.expander(" Example Scenarios - Try These!"):
    st.markdown("""
    ### High Income Example (Expected: >$50K)
    - **Age:** 45
    - **Education:** 16 years (Master's degree)
    - **Work Class:** Private
    - **Occupation:** Exec-managerial
    - **Hours/Week:** 50
    - **Marital Status:** Married-civ-spouse
    - **Capital Gain:** $15,000
    - **Sex:** Male
    - **Race:** White
    
    ---
    
    ### Low Income Example (Expected: ≤$50K)
    - **Age:** 25
    - **Education:** 10 years (Some high school)
    - **Work Class:** Private
    - **Occupation:** Other-service
    - **Hours/Week:** 35
    - **Marital Status:** Never-married
    - **Capital Gain:** $0
    - **Sex:** Female
    - **Race:** White
    
    ---
    
    ### Mid-Level Example (Uncertain)
    - **Age:** 38
    - **Education:** 13 years (Bachelor's degree)
    - **Work Class:** Private
    - **Occupation:** Tech-support
    - **Hours/Week:** 40
    - **Marital Status:** Divorced
    - **Capital Gain:** $0
    - **Sex:** Male
    - **Race:** Asian-Pac-Islander
    """)

#Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p><strong>Income Level Predictor</strong> | Built with Streamlit | Model: {}</p>
    <p>Based on US Census Adult Income dataset | F1-Score: {:.3f}</p>
</div>
""".format(metadata['model_name'], metadata['f1_score']), unsafe_allow_html=True)
