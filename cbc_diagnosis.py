
import json
import pandas as pd
import cv2
import pytesseract
import re
import numpy as np
import random
import pickle

from sklearn.preprocessing import StandardScaler
from cbc import CBCDataProcessor

cbc = CBCDataProcessor()
path = r"test_image3.jpg"
df = cbc.main(path)
pd.set_option('display.max_columns', None)

df.columns

pickle_file_path = 'cbc_scaler_2.pkl'

# Load the pickle file
with open(pickle_file_path, 'rb') as file:
    scaler_model = pickle.load(file)

pickle_file_path = 'cbc_classify.pkl'

# Load the pickle file
with open(pickle_file_path, 'rb') as file:
    classifier = pickle.load(file)


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
       'MCH', 'MCHC', 'RDW', 'MPV', 'Mentezer Index', 'Retic Count', 'ESR',
       'CRP']

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



processor = preprocess_cbc_data()
processed_df = processor.main(df)
new_data_scaled = scaler_model.transform(processed_df)
preds = classifier.predict_proba(new_data_scaled)



# Get the indices of the top 3 probabilities for each sample
top_3_indices = np.argsort(preds, axis=1)[:, -3:]

# Get the top 3 probabilities for each sample
top_3_probs = np.take_along_axis(preds, top_3_indices, axis=1)

# Print the top 3 probabilities and their corresponding indices
for i, (probs, indices) in enumerate(zip(top_3_probs, top_3_indices)):
    print(f"Sample {i}:")
    
    # Print top 3 probabilities and their corresponding class indices
    for prob, idx in zip(probs, indices):
        print(f"Class {idx}: {prob:.4f}")
    
    # Find and print the maximum probability for this sample
    max_prob = np.max(preds[i])
    max_prob_class = np.argmax(preds[i])
    print(f"Max Probability: Class {max_prob_class}, Probability: {max_prob:.4f}")
    print()

labels = {0: 'Acute Lymphoblastic Leukemia',
 1: 'Allergic Reactions',
 2: 'Alpha Thalassemia',
 3: 'Aplastic Anemia',
 4: 'Babesiosis',
 5: 'Bacterial Infection',
 6: 'Beta Thalassemia',
 7: 'Celiac Disease',
 8: 'Chronic Inflammation',
 9: 'Chronic Kidney Disease',
 10: 'Chronic Myeloid Leukemia',
 11: 'Folate Deficiency Anemia',
 12: 'Healthy',
 13: 'Hemochromatosis',
 14: 'Hemolytic Anemia',
 15: 'Leptospirosis',
 16: 'Lymphatic Filariasis',
 17: 'Lymphoma',
 18: 'Malaria',
 19: 'Megaloblastic Anemia',
 20: 'Microfilaria Infection',
 21: 'Mononucleosis',
 22: 'Multiple Myeloma',
 23: 'Osteomyelitis',
 24: 'Pernicious Anemia',
 25: 'Polycythemia Vera',
 26: 'Polymyalgia Rheumatica',
 27: 'Rheumatic Fever',
 28: 'Sarcoidosis',
 29: 'Sepsis',
 30: 'Sickle Cell Disease',
 31: 'Systemic Lupus Erythematosus',
 32: 'Systemic Vasculitis',
 33: 'Temporal Arteritis',
 34: 'Thrombocytopenia',
 35: 'Thrombocytosis',
 36: 'Tuberculosis',
 37: 'Viral Infection',
 38: 'Vitamin B12 Deficiency Anemia'}

if max_prob_class in labels.keys():
    final_result = labels[max_prob_class]

import json

def output_simplifier(df, final_result):
    # Fill NaN values with 'None'
    df_output = df.fillna('None')

    # Create predictions dictionary
    predictions = {
        'Age': df_output['Age'].tolist(),
        'Haemoglobin Level': df_output['Haemoglobin Level'].tolist(),
        'R.B.C Count': df_output['R.B.C Count'].tolist(),
        'W.B.C Count': df_output['W.B.C Count'].tolist(),
        'Platelets Count': df_output['Platelets Count'].tolist(),
        'Neutrophils': df_output['Neutrophils'].tolist(),
        'Lymphocytes': df_output['Lymphocytes'].tolist(),
        'Eosinophils': df_output['Eosinophils'].tolist(),
        'Monocytes': df_output['Monocytes'].tolist(),
        'Basophils': df_output['Basophils'].tolist(),
        'HCT': df_output['HCT'].tolist(),
        'WBC Morphology': df_output['WBC Morphology'].tolist(),
        'Monocyte Morphology': df_output['Monocyte Morphology'].tolist(),
        'RBC Shape': df_output['RBC Shape'].tolist(),
        'Blood Parasites': df_output['Blood Parasites'].tolist(),
        'PCV': df_output['PCV'].tolist(),
        'MCV': df_output['MCV'].tolist(),
        'MCH': df_output['MCH'].tolist(),
        'MCHC': df_output['MCHC'].tolist(),
        'RDW': df_output['RDW'].tolist(),
        'MPV': df_output['MPV'].tolist(),
        'Mentezer Index': df_output['Mentezer Index'].tolist(),
        'Retic Count': df_output['Retic Count'].tolist(),
        'ESR': df_output['ESR'].tolist(),
        'Diagnosis': final_result
    }

    # Save predictions to a JSON file
    with open('cbc_output.json', 'w') as json_file:
        json.dump(predictions, json_file, indent=4)

    # Return the output dictionary
    return predictions

output = output_simplifier(df, final_result)
print(output)
