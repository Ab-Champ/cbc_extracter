import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import json
from cbc import CBCDataProcessor

# Define the labels for disease classification
labels = {
    0: 'Acute Lymphoblastic Leukemia', 1: 'Allergic Reactions', 2: 'Alpha Thalassemia', 3: 'Aplastic Anemia',
    4: 'Babesiosis', 5: 'Bacterial Infection', 6: 'Beta Thalassemia', 7: 'Celiac Disease', 8: 'Chronic Inflammation',
    9: 'Chronic Kidney Disease', 10: 'Chronic Myeloid Leukemia', 11: 'Folate Deficiency Anemia', 12: 'Healthy',
    13: 'Hemochromatosis', 14: 'Hemolytic Anemia', 15: 'Leptospirosis', 16: 'Lymphatic Filariasis', 17: 'Lymphoma',
    18: 'Malaria', 19: 'Megaloblastic Anemia', 20: 'Microfilaria Infection', 21: 'Mononucleosis', 22: 'Multiple Myeloma',
    23: 'Osteomyelitis', 24: 'Pernicious Anemia', 25: 'Polycythemia Vera', 26: 'Polymyalgia Rheumatica',
    27: 'Rheumatic Fever', 28: 'Sarcoidosis', 29: 'Sepsis', 30: 'Sickle Cell Disease', 31: 'Systemic Lupus Erythematosus',
    32: 'Systemic Vasculitis', 33: 'Temporal Arteritis', 34: 'Thrombocytopenia', 35: 'Thrombocytosis', 36: 'Tuberculosis',
    37: 'Viral Infection', 38: 'Vitamin B12 Deficiency Anemia'
}

# Cache the model loading to avoid re-loading on every rerun
@st.cache_resource
def load_models():
    """Load the scaler and classifier models and cache them."""
    with open('cbc_scaler_2.pkl', 'rb') as f:
        scaler_model = pickle.load(f)

    with open('cbc_classify.pkl', 'rb') as f:
        classifier = pickle.load(f)

    return scaler_model, classifier

# Cache the image processing step
@st.cache_data
def process_image(image_path):
    """Process the image to extract CBC data and cache the result."""
    cbc = CBCDataProcessor()  
    df = cbc.main(image_path)  # Get the CBC dataframe from the image using your defined logic
    return df

# Preprocessing step with caching
@st.cache_data
def preprocess_cbc_data(df):
    """Preprocess the extracted CBC dataframe and cache the result."""
    true_numeric = [
        'Age', 'Haemoglobin Level', 'R.B.C Count', 'W.B.C Count', 'Platelets Count', 'Neutrophils', 'Lymphocytes',
        'Eosinophils', 'Monocytes', 'Basophils', 'Absolute Neutrophils', 'Absolute Lymphocytes', 'Absolute Eosinophils',
        'Absolute Monocytes', 'Absolute Basophils', 'HCT', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'MPV', 'Mentezer Index',
        'Retic Count', 'ESR', 'CRP'
    ]

    # Drop 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)

    # Fill missing numeric columns with float type
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    missing_columns = [col for col in true_numeric if col not in numeric_cols]

    for col in missing_columns:
        df[col] = 0.0

    # Process 'R.B.C Count' if it's a string
    if df['R.B.C Count'].dtype == 'object':
        df['R.B.C Count'] = df['R.B.C Count'].replace('Normal', round(random.uniform(4.0, 6.0)))
        df['R.B.C Count'] = df['R.B.C Count'].astype(float)

    # One-hot encode categorical columns and reindex to ensure consistency
    categorical = df.select_dtypes(include='object')
    cat_enc = pd.get_dummies(categorical, dtype=int)
    
    # Define expected categories
    all_categories = [
        'Sex_Female', 'Sex_Male', 'WBC Morphology_Giant Cells', 'WBC Morphology_Hypersegmented Neutrophils',
        'WBC Morphology_Monoblasts', 'WBC Morphology_Normal', 'WBC Morphology_Toxic Granulation',
        'Monocyte Morphology_Normal', 'RBC Shape_Anisocytosis', 'RBC Shape_Elliptical', 'RBC Shape_Macrocytic',
        'RBC Shape_NORMOCHROMIC,NORMOCYTIC', 'RBC Shape_Sickle-Shaped', 'RBC Shape_Teardrop',
        'Blood Parasites_Babesia spp.', 'Blood Parasites_Microfilaria', 'Blood Parasites_No Data',
        'Blood Parasites_Plasmodium', 'Blood Parasites_Wuchereria bancrofti'
    ]
    cat_enc = cat_enc.reindex(columns=all_categories, fill_value=0)

    # Combine numeric and categorical DataFrames
    df = pd.concat([df.select_dtypes(include=['int', 'float']), cat_enc], axis=1)

    # Drop unnecessary columns for prediction
    columns_to_drop = [
        'Lymphocytes', 'Basophils', 'Absolute Neutrophils', 'Absolute Lymphocytes', 'Absolute Eosinophils',
        'Absolute Monocytes', 'Absolute Basophils', 'MCHC', 'MPV', 'Mentezer Index', 'CRP'
    ]
    df.drop(columns=columns_to_drop, axis=1, errors='ignore', inplace=True)

    # Fill NaN values with 0
    df.fillna(0, inplace=True)

    return df

# Prediction function with caching
@st.cache_data
def predict_disease(df, scaler, classifier, labels):
    """Predict disease probabilities and return top results."""
    # Scale the data
    new_data_scaled = scaler.transform(df)

    # Get predictions
    preds = classifier.predict_proba(new_data_scaled)

    # Get top 3 predictions
    top_3_indices = np.argsort(preds, axis=1)[:, -3:]
    top_3_probs = np.take_along_axis(preds, top_3_indices, axis=1)

    # Create a dictionary of results
    results = {}
    for i, (probs, indices) in enumerate(zip(top_3_probs, top_3_indices)):
        results[f'Sample {i+1}'] = [{"Class": labels[idx], "Probability": f"{prob:.4f}"} for prob, idx in zip(probs, indices)]

    return results

# Main Streamlit App
def main():
    st.title("CBC Image Classification")

    # Upload image
    uploaded_file = st.file_uploader("Upload a CBC Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded image to a temporary file
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.read())

        # Display the uploaded image
        st.image("temp_image.jpg", caption="Uploaded Image", use_column_width=True)

        # Load models
        scaler, classifier = load_models()

        # Process the image and extract data
        df = process_image("temp_image.jpg")
        st.write("Extracted CBC Data:")
        st.write(df)  # Display extracted data

        # Preprocess the data
        processed_df = preprocess_cbc_data(df)

        # Get predictions
        results = predict_disease(processed_df, scaler, classifier, labels)

        # Display results
        st.write("Prediction Results:")
        for sample, predictions in results.items():
            st.write(f"{sample} Predictions:")
            for prediction in predictions:
                st.write(f"{prediction['Class']}: {prediction['Probability']}")

if __name__ == "__main__":
    main()
