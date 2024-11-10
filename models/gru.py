#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Masking, InputLayer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from os import listdir
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Masking, InputLayer
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.base import BaseEstimator
import tensorflow as tf

# Local Path
data_path = r'..\data'

# Files to Read
data_files = [i for i in listdir(data_path) if 'model' in i]

# Read and concat data
data = pd.concat([pd.read_csv(data_path + '\\' + file) for file in data_files])

# for column in data.columns:
#     data[column + '_was_missing'] = data[column].isnull().astype(int)

# data = data.fillna(0)
data = data.fillna(method='bfill').fillna(method='ffill')

# Sort and group data by 'Unique Stay' and 'sequence_num' to ensure time order
data = data.sort_values(by=['Unique Stay', 'sequence_num'])

# Create sequences for each 'Unique Stay'
unique_stays = data['Unique Stay'].unique()
sequences = []
labels = []

for stay in unique_stays:
    stay_data = data[data['Unique Stay'] == stay]
    
    # Drop non-feature columns and extract features as a numpy array
    features_columns = ['anchor_age', 'Arterial Blood Pressure mean', 'Arterial O2 pressure', 
                          'Creatinine (serum)', 'Dobutamine', 'Dopamine', 'Epinephrine', 
                          'Heart Rate', 'Inspired O2 Fraction', 'Lactic Acid', 
                          'Norepinephrine', 'Platelet Count', 'Total Bilirubin']

    # features_columns.extend([i + '_was_missing' for i in features_columns])

    features = stay_data[features_columns].values
    
    # Target label: use the mortality status from the last row of each stay
    label = stay_data['mortality'].values[-1]  # Assuming 'mortality' is the target
    
    # Add the sequence and corresponding label
    sequences.append(features)
    labels.append(label)

# Pad sequences to have uniform length
max_seq_length = 50  # You can adjust this based on your data distribution
sequences_padded = pad_sequences(sequences, maxlen=max_seq_length, dtype='float32', padding='post', truncating='post')

# Convert labels to numpy array
labels = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(sequences_padded, labels, test_size=0.2, random_state=42)

# Define custom wrapper for the Keras model to use with RandomizedSearchCV
class KerasModelWrapper(BaseEstimator):
    def __init__(self, learning_rate=0.001, gru_units=64, dropout_rate=0.2, epochs=10, batch_size=32):
        self.learning_rate = learning_rate
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def create_model(self):
        model = Sequential([
            InputLayer(input_shape=(X_train.shape[1], X_train.shape[2])), 
            Masking(mask_value=0.0),
            GRU(self.gru_units, return_sequences=False, dropout=self.dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='binary_crossentropy', metrics=['accuracy',
                                                                                                            tf.keras.metrics.Precision(name='precision'), 
                                                                                                            tf.keras.metrics.Recall(name='recall'),
                                                                                                            tf.keras.metrics.AUC(name='auc')])
        return model

    def fit(self, X, y):
        self.model = self.create_model()
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]

# Instantiate the model wrapper
model = KerasModelWrapper()

# Define the parameter grid for tuning
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'gru_units': [32, 64, 128],
    'dropout_rate': [0.2, 0.3, 0.5],
    'batch_size': [16, 32, 64],
    'epochs': [10, 20]
}

# RandomizedSearchCV to tune hyperparameters
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, cv=3, verbose=2, n_jobs=-1, random_state=42)

# Perform the search
random_search_result = random_search.fit(X_train, y_train)

# Best parameters and score
print("Best Hyperparameters:", random_search_result.best_params_)
print("Best Accuracy:", random_search_result.best_score_)

# Evaluate the best model on the test set
best_model = random_search_result.best_estimator_
test_acc = best_model.score(X_test, y_test)
print(f'Test Accuracy: {test_acc:.4f}')

# Save the best model
best_model = random_search_result.best_estimator_

# Save the best model to a file
model_save_path = 'gru_best_model.keras'
# best_model.model.save(model_save_path)

print(f"Best model saved at: {model_save_path}")

# Evaluate the best model on the test set
test_acc = best_model.score(X_test, y_test)
print(f'Test Accuracy: {test_acc:.4f}')

# Load the saved model and re-evaluate on the test set
loaded_model = load_model(model_save_path)
model_metrics = loaded_model.evaluate(X_test, y_test, verbose=0)

