#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd

class CBCDataProcessor:
    import cv2
    import pytesseract
    import re
    import pandas as pd
    
    def __init__(self):
        self.data = {
            'Patient Name': 'NaN',
            'Patient ID': 'NaN',
            'Doctor': 'NaN',
            'Age': 'NaN',
            'Sex': 'NaN',
            'Haemoglobin Level': 'NaN',
            'R.B.C Count': 'NaN',
            'W.B.C Count': 'NaN',
            'Platelets Count': 'NaN',
            'Neutrophils': 'NaN',
            'Lymphocytes': 'NaN',
            'Eosinophils': 'NaN',
            'Monocytes': 'NaN',
            'Basophils': 'NaN',
            'Absolute Neutrophils': 'NaN',
            'Absolute Lymphocytes': 'NaN',
            'Absolute Eosinophils': 'NaN',
            'Absolute Monocytes': 'NaN',
            'Absolute Basophils': 'NaN',
            'HCT': 'NaN',
            'WBC Morphology': 'NaN',
            'Monocyte Morphology': 'NaN',
            'RBC Shape': 'NaN',
            'Blood Parasites': 'NaN',
            'PCV': 'NaN',
            'MCV': 'NaN',
            'MCH': 'NaN',
            'MCHC': 'NaN',
            'RDW': 'NaN',
            'MPV': 'NaN',
            'Mentezer Index': 'NaN',
            'Retic Count': 'NaN',
            'ESR': 'NaN',
        }

        self.parameter_mapping = {
            'age': 'Age',
            'gender': 'Sex',
            'sex': 'Sex',
            'haemoglobin level': 'Haemoglobin Level',
            'hemoglobin': 'Haemoglobin Level',
            '(hb)': 'Haemoglobin Level',
            'rbc': 'R.B.C Count',
            'red blood cell count': 'R.B.C Count',
            '(rbc)': 'R.B.C Count',
            'wbc': 'W.B.C Count',
            'white blood cell count': 'W.B.C Count',
            'total wbc count': 'W.B.C Count',
            'platelets count': 'Platelets Count',
            'platelets' : 'Platelets Count',
            'platelet': 'Platelets Count',
            'neutrophils': 'Neutrophils',
            'lymphocytes': 'Lymphocytes',
            'eosinophils': 'Eosinophils',
            'monocytes': 'Monocytes',
            'basophils': 'Basophils',
            'hct': 'HCT',
            'hematocrit': 'HCT',
            'pcv': 'PCV',
            'packed cell volume': 'PCV',
            '(pcv)': 'PCV',
            'mcv': 'MCV',
            '(mcv)': 'MCV',
            'mch': 'MCH',
            '(mch)': 'MCH',
            'mchc': 'MCHC',
            '(mchc)': 'MCHC',
            'rdw': 'RDW',
            'mpv': 'MPV',
            'mentezer index': 'Mentezer Index',
            'retic count': 'Retic Count',
            'esr': 'ESR',
            'erythrocyte sedimentation rate': 'ESR',
            'rbc morphology': 'RBC Shape',
            'red cell morphology': 'RBC Shape',
            'wbc morphology': 'WBC Morphology',
            'monocyte morphology': 'Monocyte Morphology',
            'blood parasites': 'Blood Parasites',
            'absolute neutrophils': 'Absolute Neutrophils',
            'absolute lymphocytes': 'Absolute Lymphocytes',
            'absolute eosinophils': 'Absolute Eosinophils',
            'absolute monocytes': 'Absolute Monocytes',
            'absolute basophils': 'Absolute Basophils'
        }
        self.expected_ranges = {
            'Haemoglobin Level': (0, 25),  # g/dL
            'R.B.C Count': (0, 10),  # mill/cumm
            'W.B.C Count': (0, 20000),  # cumm
            'Platelets Count': (0, 1000000),  # cumm
            'Neutrophils': (0, 100),  # percentage
            'Lymphocytes': (0, 100),  # percentage
            'Eosinophils': (0, 100),  # percentage
            'Monocytes': (0, 100),  # percentage
            'Basophils': (0, 100),  # percentage
            'Absolute Neutrophils': (0, 10000),  # cells/mL
            'Absolute Lymphocytes': (0, 5000),  # cells/mL
            'Absolute Eosinophils': (0, 1000),  # cells/mL
            'Absolute Monocytes': (0, 1000),  # cells/mL
            'Absolute Basophils': (0, 500),  # cells/mL
            'PCV': (0, 100),  # percentage
            'MCV': (0, 150),  # fL
            'MCH': (0, 50),  # pg
            'MCHC': (0, 40),  # g/dL
            'RDW': (0, 30),  # percentage
            'MPV': (0, 15),  # fL
            'Mentezer Index': (0, 50),  # ratio
            'Retic Count': (0, 100),  # percentage
            'ESR': (0, 100),  # mm/hr
        } 


    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Increase contrast and sharpen image
        gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
        # Apply Gaussian blur and adaptive thresholding to binarize image
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return binary_image
    
    def extract_text_from_image(self, processed_image):
        custom_config = r'--oem 3 --psm 6'  # PSM 6 assumes a single uniform block of text
        extracted_text = pytesseract.image_to_string(processed_image, config=custom_config)
        return extracted_text
    
    
    def process_image_main(self, image_path):
        processed_image = self.preprocess_image(image_path)
        extracted_text = self.extract_text_from_image(processed_image)
        return extracted_text


    def line_process(self, ocr_text):
        """
        Splits the OCR text into lines and then into words.
        """
        lines = ocr_text.splitlines()
        line_list = []
        for line in lines:
            # Split on whitespace and handle multiple spaces
            line_list.append(re.split(r'\s+', line.strip()))
        return line_list
    
    def merge_and_clean_elements(self, lst):
        """
        Merges and cleans elements from a line split by whitespace.
        """
        merged_list = []
        i = 0
        while i < len(lst):
            current_element = lst[i]
            
            # Merge "ABSOLUTE" (case insensitive) with the next element
            if current_element.upper() == "ABSOLUTE" and i + 1 < len(lst):
                next_element = lst[i + 1]
                current_element += " " + next_element  # Merge with the next element
                merged_list.append(current_element)  # Append the merged element
                i += 1  # Skip the next element since it's already merged
    
            elif re.search(r'\d+[-/]\d+', current_element):
                # Join numeric ranges separated by '-' or '/'
                current_element = re.sub(r'[^\d\-\/]', '', current_element)
                merged_list.append(current_element)
            else:
                # Clean and merge numeric parts with alphabetic characters
                if re.search(r'\d', current_element):
                    current_element = re.sub(r'[^\d\.\-]', '', current_element)
                
                # Append and handle special cases
                merged_list.append(current_element)
            
            i += 1
        
        return merged_list


    def extract_and_split_numeric_parts(self, lst):
        """
        Splits elements containing numeric ranges and cleans them.
        """
        cleaned_list = []
    
        for sublist in lst:
            new_sublist = []
            for item in sublist:
                # Handle numeric ranges and clean them
                if '-' in item and any(char.isdigit() for char in item):
                    item = re.sub(r'[^\d\-]', '', item)
                    split_items = item.split('-')
                    new_sublist.extend(split_items)
                else:
                    # Clean individual items
                    cleaned_item = re.sub(r'[^\d\.\-]', '', item)
                    if cleaned_item:
                        new_sublist.append(cleaned_item)
                    else:
                        new_sublist.append(item)
    
            cleaned_list.append(new_sublist)
        return cleaned_list
    
    def filter_unnecessary_text(self, lst):
        """
        Filters out unnecessary lists and single character elements.
        """
        filtered_list = []
        for sublist in lst:
            # Remove lists with irrelevant text
            if any(re.search(r'[A-Za-z]+', item) for item in sublist):
                # Remove single character elements and symbols
                cleaned_sublist = [item for item in sublist if not re.fullmatch(r'[^\w\s]', item)]
                if cleaned_sublist:
                    filtered_list.append(cleaned_sublist)
        
        return filtered_list
    
    def process_extracted_ocr(self, ocr_text):
        """
        Processes the raw OCR text to extract structured information.
        """
        data = self.line_process(ocr_text)
        
        # Remove sublists containing 'Calculated' or 'calculated'
        filtered_data = [sublist for sublist in data if 'Calculated' not in sublist and 'calculated' not in sublist]
        # Merge and clean elements
        merged_list = [self.merge_and_clean_elements(sublist) for sublist in filtered_data]
        # Extract and split numeric parts
        cleaned_ocr = self.extract_and_split_numeric_parts(merged_list)
        final_cleaned_ocr = self.filter_unnecessary_text(cleaned_ocr)
        
        return final_cleaned_ocr

    def extract_data(self, data_block):
        for row in data_block:
            lower_row = [element.lower() for element in row]

            if 'patient id' in lower_row or 'id' in lower_row:
                try:
                    index = lower_row.index('id') if 'id' in lower_row else lower_row.index('patient id')
                    self.data['Patient ID'] = row[index + 1]
                except IndexError:
                    self.data['Patient ID'] = None

            if 'patient name' in lower_row or 'name' in lower_row:
                try:
                    index = lower_row.index('name') if 'name' in lower_row else lower_row.index('patient name')
                    self.data['Patient Name'] = ' '.join(row[index + 1: index + 3])
                except IndexError:
                    self.data['Patient Name'] = None

            if 'doctor' in lower_row:
                try:
                    index = lower_row.index('doctor')
                    self.data['Doctor'] = ' '.join(row[index + 1: index + 3])
                except IndexError:
                    self.data['Doctor'] = None

            for key, value in self.parameter_mapping.items():
                if key in lower_row:
                    index = lower_row.index(key)

                    if index + 1 < len(row) and row[index + 1].replace('.', '', 1).isdigit():
                        self.data[value] = row[index + 1]
                    elif index + 2 < len(row) and row[index + 2].replace('.', '', 1).isdigit():
                        self.data[value] = row[index + 2]

                    if key in ['wbc morphology', 'monocyte morphology', 'rbc morphology', 'blood parasites']:
                        self.data[value] = row[-1]

                    if 'count' in key:
                        try:
                            self.data[value] = next((row[i+1] for i, val in enumerate(lower_row) if val == key and i+1 < len(row) and row[i+1].isdigit()), None)
                        except StopIteration:
                            pass

                    if value == 'Sex' and ('female' in lower_row or 'male' in lower_row):
                        self.data['Sex'] = 'FEMALE' if 'female' in lower_row else 'MALE'

            if 'rbc morphology' in " ".join(lower_row) or 'red cell morphology' in " ".join(lower_row):
                self.data['RBC Shape'] = row[-1]

            for absolute_key in ['absolute neutrophils', 'absolute lymphocytes', 'absolute eosinophils', 'absolute monocytes', 'absolute basophils']:
                if absolute_key in lower_row:
                    self.data[self.parameter_mapping[absolute_key]] = 'NaN'

        return self.data

    def process_dataframe(self):
        df = pd.DataFrame([self.data])
        numeric_cols = ['Haemoglobin Level', 'R.B.C Count', 'W.B.C Count', 'Platelets Count',
                        'Neutrophils', 'Lymphocytes', 'Eosinophils', 'Monocytes', 'Basophils',
                        'Absolute Neutrophils', 'Absolute Lymphocytes', 'Absolute Eosinophils',
                        'Absolute Monocytes', 'Absolute Basophils', 'HCT', 'PCV', 'MCV',
                        'MCH', 'MCHC', 'RDW', 'MPV', 'Mentezer Index', 'Retic Count', 'ESR']

        for col in numeric_cols:
            df[col] = df[col].astype('float')

        if 'W.B.C Count' in df.columns:
            wbc_count = df['W.B.C Count'].iloc[0]
            for key in ['Neutrophils', 'Lymphocytes', 'Eosinophils', 'Monocytes', 'Basophils']:
                if key in df.columns and pd.notna(df[key].iloc[0]) and pd.notna(wbc_count):
                    percentage = df[key].iloc[0]
                    absolute_key = 'Absolute ' + key
                    df[absolute_key] = round((percentage / 100) * wbc_count, 2)
                else:
                    df['Absolute ' + key] = 'NaN'
                    
        return df

    def replace_out_of_range_with_none(self, df):
        for column_name, (min_value, max_value) in self.expected_ranges.items():
            if column_name in df.columns:
                df[column_name] = df[column_name].apply(
                    lambda x: float(x) if pd.api.types.is_numeric_dtype(df[column_name]) and min_value <= float(x) <= max_value else None
                )
                
        return df


    def structure_main(self, ocr):
        self.extract_data(ocr)
        df = self.process_dataframe()
        df = self.replace_out_of_range_with_none(df)
        return df

    #All processing Function
    def main(self,image_path):
        ocr = self.process_image_main(image_path)
        cleaned_ocr = self.process_extracted_ocr(ocr)
        processed_df = self.structure_main(cleaned_ocr)
        return processed_df


