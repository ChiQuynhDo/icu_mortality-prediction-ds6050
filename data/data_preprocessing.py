#%%

import pandas as pd
from os import listdir
import numpy as np

root = r'.\datasets'

data = {file.replace('.csv', ''): pd.read_csv(root + '\\' + file) for file in listdir(root)}

#%%
## Look at tables available
# print('HOSP TABLES:')
# for file in hosp_data:
#     print(file, hosp_data[file].shape)

# print('------')
# print('ICU TABLES:')
# for file in icu_data:
#     print(file, icu_data[file].shape)


# Only get columns needed
admissions = data['admissions'][['subject_id','hadm_id', 'insurance', 'race', 'marital_status']]

# Only get columns needed
patient = data['patients'][['subject_id', 'gender', 'anchor_age', 'dod']]

# Only get columns needed
d_items = data['icu_d_items'][['itemid', 'label']]

# Only get columns needed
input_events = data['events'][['subject_id', 'hadm_id', 'itemid', 'endtime', 'amount', 'amountuom', 'patientweight']]

# Merge D_ITEMS
input_events = input_events.merge(d_items, how='left', on='itemid')

input_kept_features =  ['Dopamine', 'Dobutamine', 'Epinephrine.', 'Epinephrine', 'Norepinephrine']

# Query for Relevant Features
input_events = input_events.query(f"label in {input_kept_features}")

# Aggregated multiple data sets
chart_names = ['chart_events1', 'chart_events2', 'chart_events3']

# Only get columns needed
chart_events = pd.concat([data[i] for i in chart_names])[['subject_id', 'hadm_id', 'itemid', 'charttime', 'value', 'valuenum', 'valueuom']]

# Merge D_ITEMS
chart_events = chart_events.merge(d_items, how='left', on='itemid')

chart_kept_features = ['Arterial Blood Pressure mean', 'Heart Rate', 'GU Irrigant/Urine Volume Out', 'Urine and GU Irrigant Out', 'PO2_ApacheIV', 'Arterial O2 pressure', 'Inspired O2 Fraction', 'Lactic Acid', 'Platelet Count', 'Total Bilirubin', 'Creatinine (serum)', 'Ventilator Type', 'Ventilator Mode']

# Query for Relevant Features
chart_events = chart_events.query(f"label in {chart_kept_features}")

## Prep Chart Events and Input Events for Concatentation

# Isolate patient weight
patient_weight = input_events[['subject_id', 'hadm_id', 'patientweight']].groupby(by=['subject_id', 'hadm_id'])['patientweight'].mean().reset_index()

# Rename columns to have like naming conventions
chart_events.rename(columns={'charttime': 'endtime', 'valuenum': 'amount', 'valueuom': 'amountuom'}, inplace=True)

# Drop Patient Weight since there is a new table for it 
input_events.drop(columns=['patientweight'], inplace=True)

# Concatenate
events = pd.concat([chart_events, input_events])

# Drop itemid since we already merged the items
events.drop(columns=['itemid'], inplace=True)

# Merge patient to admission
df = admissions.merge(patient, how='left', on='subject_id')

# Merge events to full data
df = df.merge(events, how='inner', on=['subject_id', 'hadm_id'])

# Binary classification for Date of Death
df['Died'] = df['dod'].notnull()

# Get Unique Stay Identifier
df['Unique Stay'] = df['subject_id'].astype(str) + df['hadm_id'].astype(str)

# Only keep values for Ventilator Mode and Type
df['value'] = np.where(df['label'].isin(['Ventilator Mode', 'Ventilator Type']), df['value'], np.nan)

# Convert to Time Stamp
df['endtime'] = pd.to_datetime(df['endtime'])

# Get Sequence Num
df = df.sort_values(by=['Unique Stay', 'endtime']).reset_index(drop=True)

df['sequence_num'] = df.groupby('Unique Stay').cumcount() + 1

# One Hot Encode Categorical
# df = pd.get_dummies(df, columns=['race', 'insurance', 'marital_status', 'gender', 'amountuom', 'label', 'value'])

# Drop Time of Death Column and Identifier Columns
df.drop(columns=['dod', 'subject_id', 'hadm_id'], inplace=True)

# Save
# df.to_csv('model_data.csv', index=False)
#%%
for i, chunk in enumerate(range(0, df.shape[0], 100000)):
    # Creating file names dynamically based on chunk number
    chunk_df = df.iloc[chunk:chunk + 100000]
    chunk_df.to_csv(f'final_data\model_data_chunk_{i+1}.csv', index=False)