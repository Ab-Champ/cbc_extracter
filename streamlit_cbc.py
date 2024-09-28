import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
from PIL import Image
import cv2
import pytesseract
from cbc import CBCDataProcessor

# Function to preprocess the image and get predictions
class preprocess_cbc_data:
    def __init__(self):
        self.scaler = StandardScaler()

    def main(self, data):
        data_df = data
        
        # Drop 'Unnamed: 0' column if it exists
        if 'Unnamed: 0' in data_df.columns:
            data_df.drop('Unnamed: 0', axis=1, inplace=True)

        true_numeric = ['Age', 'Haemoglobin Level', 'R.B.C Count', 'W.B.C Count', 'Platelets Count',
                        'Neutrophils', 'Lymphocytes', 'Eosinophils', 'Monocytes', 'Basophils',
                        'Absolute Neutrophils', 'Absolute Lymphocytes', 'Absolute Eosinophils',
                        'Absolute Monocytes', 'Absolute Basophils', 'HCT','PCV', 'MCV',
                        'MCH', 'MCHC', 'RDW', 'MPV', 'Mentezer Index', 'Retic Count', 'ESR', 'CRP']

        numeric = data_df.select_dtypes(include=['int', 'float'])

        missing_columns = [col for col in true_numeric if col not in numeric.columns]

        if missing_columns:
            for col in missing_columns:
                data_df[col]=data_df[col].astype('float')

        # Process 'R.B.C Count'
        if data_df['R.B.C Count'].dtype == 'object':
            data_df['R.B.C Count'] = data_df['R.B.C Count'].replace('Normal', round(random.uniform(4.0, 6.0)))
            data_df['R.B.C Count'] = data_df['R.B.C Count'].astype(float)

        # One-hot encode categorical columns
        categorical = data_df.select_dtypes(include='object')
        cat_enc = pd.get_dummies(categorical, dtype=int) 

        # Numeric columns
        numeric = data_df.select_dtypes(include=['int', 'float'])

        # Define expected categorical columns
        all_categories = ['Sex_Female', 'Sex_Male',
                          'WBC Morphology_Giant Cells',
                          'WBC Morphology_Hypersegmented Neutrophils',
                          'WBC Morphology_Monoblasts', 'WBC Morphology_Normal',
                          'WBC Morphology_Toxic Granulation', 'Monocyte Morphology_Normal',
                          'RBC Shape_Anisocytosis', 'RBC Shape_Elliptical',
                          'RBC Shape_Macrocytic', 'RBC Shape_NORMOCHROMIC,NORMOCYTIC',
                          'RBC Shape_Sickle-Shaped', 'RBC Shape_Teardrop',
                          'Blood Parasites_Babesia spp.', 'Blood Parasites_Microfilaria',
                          'Blood Parasites_No Data', 'Blood Parasites_Plasmodium',
                          'Blood Parasites_Wuchereria bancrofti']

        # Reindex one-hot encoded DataFrame to include all expected categories
        cat_enc = cat_enc.reindex(columns=all_categories, fill_value=0)

        # Combine numeric and categorical DataFrames
        df = pd.concat([numeric, cat_enc], axis=1)

        columns=['Lymphocytes', 'Basophils', 'Absolute Neutrophils', 'Absolute Lymphocytes',
                 'Absolute Eosinophils', 'Absolute Monocytes', 'Absolute Basophils', 'MCHC',
                 'MPV', 'Mentezer Index', 'CRP']

        df_new = df.drop(columns, axis=1)

        df_new.fillna(0, inplace=True)
        return df_new

# Labels for classification
labels = {
    0: 'Acute Lymphoblastic Leukemia', 1: 'Allergic Reactions', 2: 'Alpha Thalassemia',
    3: 'Aplastic Anemia', 4: 'Babesiosis', 5: 'Bacterial Infection', 6: 'Beta Thalassemia',
    7: 'Celiac Disease', 8: 'Chronic Inflammation', 9: 'Chronic Kidney Disease', 10: 'Chronic Myeloid Leukemia',
    11: 'Folate Deficiency Anemia', 12: 'Healthy', 13: 'Hemochromatosis', 14: 'Hemolytic Anemia',
    15: 'Leptospirosis', 16: 'Lymphatic Filariasis', 17: 'Lymphoma', 18: 'Malaria', 19: 'Megaloblastic Anemia',
    20: 'Microfilaria Infection', 21: 'Mononucleosis', 22: 'Multiple Myeloma', 23: 'Osteomyelitis',
    24: 'Pernicious Anemia', 25: 'Polycythemia Vera', 26: 'Polymyalgia Rheumatica', 27: 'Rheumatic Fever',
    28: 'Sarcoidosis', 29: 'Sepsis', 30: 'Sickle Cell Disease', 31: 'Systemic Lupus Erythematosus',
    32: 'Systemic Vasculitis', 33: 'Temporal Arteritis', 34: 'Thrombocytopenia', 35: 'Thrombocytosis',
    36: 'Tuberculosis', 37: 'Viral Infection', 38: 'Vitamin B12 Deficiency Anemia'
}

# Function to simplify output
def output_simplifier(df, final_result):
    df_output = df.fillna('None')
    predictions = {
        'Age': df_output['Age'].tolist(),
        'Haemoglobin Level': df_output['Haemoglobin Level'].tolist(),
        'R.B.C Count': df_output['R.B.C Count'].tolist(),
        'W.B.C Count': df_output['W.B.C Count'].tolist(),
        'Platelets Count': df_output['Platelets Count'].tolist(),
        'Neutrophils': df_output['Neutrophils'].tolist(),
        'Diagnosis': final_result
    }
    return predictions

# Streamlit App
st.title("CBC Diagnosis App")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Convert the uploaded image to an OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Load CBCDataProcessor and process the image
    cbc_processor = CBCDataProcessor()
    df = cbc_processor.main(uploaded_file.name)  # Adjust according to how your CBC processor works

    # Preprocess the data
    processor = preprocess_cbc_data()
    processed_df = processor.main(df)

    # Load the model and scaler
    with open('cbc_scaler_2.pkl', 'rb') as file:
        scaler_model = pickle.load(file)

    with open('cbc_classify.pkl', 'rb') as file:
        classifier = pickle.load(file)

    # Scale the data
    new_data_scaled = scaler_model.transform(processed_df)

    # Predict probabilities
    preds = classifier.predict_proba(new_data_scaled)

    # Get top prediction
    max_prob_class = np.argmax(preds, axis=1)[0]
    final_result = labels[max_prob_class]

    # Simplify output
    output = output_simplifier(df, final_result)

    # Display predictions
    st.write("Prediction Result:", final_result)
    st.write(output)
