import data_loader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# function to load data
def load_data(file_path):
    _train_features, train_labels, _valid_features, valid_labels, _test_features, test_labels, label_dict = data_loader.load_dataset('./dataset_diabetes/diabetic_data.csv')
    data_head = ['encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'weight', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'time_in_hospital', 'payer_code', 'medical_specialty', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed']
    #columns = diabeticData.columns
    #columns = columns[:-1]
    #last_col = diabeticData.columns[-1:]
    train_features = pd.DataFrame(data=_train_features, columns=data_head)
    valid_features = pd.DataFrame(data=_valid_features, columns=data_head)
    test_features = pd.DataFrame(data=_test_features, columns=data_head)
    
    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels, label_dict

#Impute missing values. 'UNK' represents Unknown
def process_unknowns(s_name, series):
    unknown_tokens = {
        'race': '?',
        'gender': 'Unknown/Invalid',
        'weight': '?',
        'payer_code': '?',
        'medical_specialty': '?',
        'diag_1': '?',
        'diag_2': '?',
        'diag_3': '?',
        'max_glu_serum': 'None',
        'A1Cresult': 'None',
    }
    if s_name not in unknown_tokens:
        return series
    updated_series = []
    for c in series:
        if c == unknown_tokens[s_name]:
            updated_series.append('UNK')
        else:
            updated_series.append(c)
    return pd.Series(updated_series)

#Fuction to create mapping between values and numbers
def create_mapping(values):
    mapping = {}
    rmapping = {}
    i = 1
    for k in values:
        mapping[k] = i
        rmapping[i] = k
        i += 1
    return mapping, rmapping

#Function to keep only top 10 categories
def process_column(s_name, series, top=10, mapping=None):
    all_values = dict(series.value_counts())
    
    if 'UNK' in all_values:
        all_values.pop('UNK')
        
    if len(all_values) > top:
        top_values = sorted(list(all_values.items()), reverse=True)
        top_values = top_values[:top]
        all_values = {}
        for k, v in top_values:
            all_values[k] = v
            
    all_values['UNK'] = 0
    
    rmapping = {}
    if not mapping:
        mapping, rmapping = create_mapping(all_values)
    else:
        for k, v in mapping.items():
            rmapping[v] = k
    
    total = max(mapping.values())
    final = []
    for c in series:
        t = [0]*total
        if c in mapping:
            t[mapping[c]-1] = 1
        elif 'UNK' in mapping:
            t[mapping['UNK']-1] = 1
        final.append(t)
    head = []
    for i in range(1, total+1):
        head.append('{}_{}'.format(s_name, rmapping[i]))
    return pd.DataFrame(data=final, columns=head), mapping

#Function to do preprocessing by dropping irrelevant features and converting categorical data to one-hot encoding
def process_df(df, cat_cols, explicit_drop=(), top=10, mapping=None):
    if mapping is None:
        mapping = {}
    final_df = df.copy()
    redundant_columns = ['encounter_id', 'patient_nbr', 'citoglipton', 'glimepiride-pioglitazone', 'examide']
    final_df.drop(redundant_columns, axis=1, inplace=True, errors='ignore')
    final_df.drop(explicit_drop, axis=1, inplace=True, errors='ignore')
    
    for col in cat_cols:
        series = final_df[col]
        series = process_unknowns(col, series)
        onehot_data, updated_mapping = process_column(col, series, top, mapping.get(col, None))
        mapping[col] = updated_mapping
        final_df.drop(col, axis=1, inplace=True)
        final_df = final_df.join(onehot_data)
    return final_df, mapping


# def process_labels(labels):
#     labels = pd.Series(labels, name='output')
#     labels.replace(to_replace=2, value=1, inplace=True)
#     return np.array(labels)

def preprocess():
    train_features, train_labels, valid_features, valid_labels, test_features, test_labels, label_data = load_data('./dataset_diabetes/diabetic_data.csv')

    useful_cols = ('race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'time_in_hospital',
               'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 
               'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'insulin', 'change', 'diabetesMed')

    remaining_cols = list(set(train_features.columns) - set(useful_cols))
    print("Colums to drop: {}".format(remaining_cols))
    processed_train_features, mapping = process_df(train_features, useful_cols, explicit_drop=remaining_cols)
    print("\nProcessed train features shape: {}".format(processed_train_features.shape))
    print("\nTrain labels shape: {}".format(train_labels.shape))

    processed_valid_features, _ = process_df(valid_features, useful_cols, explicit_drop=remaining_cols, mapping=mapping)
    print("\nProcessed valid features shape: {}".format(processed_valid_features.shape))
    print("\nValid labels shape: {}".format(valid_labels.shape))

    processed_test_features, _ = process_df(test_features, useful_cols, explicit_drop=remaining_cols, mapping=mapping)
    print("\nProcessed test features shape: {}".format(processed_test_features.shape))
    print("\nTest labels shape: {}".format(test_labels.shape))

    print("\nlabel_dict: {}".format(label_data))
    return processed_train_features, train_labels, processed_valid_features, valid_labels, processed_test_features, test_labels, label_data