#%%

import pandas as pd
from os import listdir

hosp_root = r'.\mimic-iv-clinical-database-demo-2.2\hosp_extract'
icu_root = r'.\mimic-iv-clinical-database-demo-2.2\icu_extract'

hosp_data = {file.replace('.csv', ''): pd.read_csv(hosp_root + '\\' + file) for file in listdir(hosp_root)}

icu_data = {file.replace('.csv', ''): pd.read_csv(icu_root + '\\' + file) for file in listdir(icu_root)}

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
admissions = hosp_data['admissions'][['subject_id','hadm_id', 'insurance', 'race', 'marital_status']]

# Only get columns needed
patient = hosp_data['patients'][['subject_id', 'gender', 'anchor_age', 'dod']]

# Only get columns needed
d_items = icu_data['d_items'][['itemid', 'label']]

# Only get columns needed
input_events = icu_data['inputevents'][['subject_id', 'hadm_id', 'itemid', 'endtime', 'storetime', 'amount', 'amountuom', 'patientweight']]

# Merge D_ITEMS
input_events = input_events.merge(d_items, how='left', on='itemid')

input_kept_features =  ['Dopamine', 'Dobutamine', 'Epinephrine.', 'Epinephrine', 'Norepinephrine']

# Query for Relevant Features
input_events = input_events.query(f"label in {input_kept_features}")

# Only get columns needed
chart_events = icu_data['chartevents'][['subject_id', 'hadm_id', 'itemid', 'charttime', 'storetime', 'valuenum', 'valueuom']]

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

# Drop Time of Death Column and Identifier Columns
df.drop(columns=['dod', 'subject_id', 'hadm_id'], inplace=True)

# Save
df.to_csv('demo_model_data.csv', index=False)
