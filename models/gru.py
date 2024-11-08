#%%
import pandas as pd
from os import listdir
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Masking, InputLayer
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.base import BaseEstimator

# Local Path
data_path = r'..\data\final_data'

# Files to Read
data_files = listdir(data_path)

# Read and concat data
data = pd.concat([pd.read_csv(data_path + '\\' + file) for file in data_files]).drop(columns=['endtime'])

# Handle categorical variables with Label Encoding
label_encoder = LabelEncoder()

# Encode categorical features
categorical_columns = ['insurance', 'race', 'marital_status', 'gender', 'value', 'amountuom', 'amount']
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col].astype(str))

# Encode target variable `Died` (binary classification: 0 = Alive, 1 = Died)
data['Died'] = label_encoder.fit_transform(data['Died'])

# Sort by Unique Stay and sequence number to get events in correct order
data.sort_values(by=['Unique Stay', 'sequence_num'], inplace=True)

# Create sequences of features for each Unique Stay
sequence_data = []
sequence_labels = []

unique_stays = data['Unique Stay'].unique()

# Iterate over unique stays
for stay in unique_stays:
    stay_data = data[data['Unique Stay'] == stay]
    stay_features = stay_data[['insurance', 'race', 'marital_status', 'gender', 'anchor_age', 'value', 'amount', 'amountuom']].values
    stay_label = stay_data['Died'].values[-1]  # Use the last event to define the label
    
    # Add the sequence and its corresponding label
    sequence_data.append(stay_features)
    sequence_labels.append(stay_label)

# Pad sequences to ensure uniform input length
sequence_data = pad_sequences(sequence_data, padding='post', dtype='float32', maxlen=50)  # Adjust maxlen as needed

# Convert labels to numpy array
sequence_labels = np.array(sequence_labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(sequence_data, sequence_labels, test_size=0.2, random_state=42)

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
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
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
    'epochs': [10, 20, 50]
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
best_model.model.save(model_save_path)

print(f"Best model saved at: {model_save_path}")

# Evaluate the best model on the test set
test_acc = best_model.score(X_test, y_test)
print(f'Test Accuracy: {test_acc:.4f}')

# Load the saved model and re-evaluate on the test set
loaded_model = load_model(model_save_path)
loaded_test_loss, loaded_test_acc = loaded_model.evaluate(X_test, y_test, verbose=0)

print(f"Loaded Model Test Accuracy: {loaded_test_acc:.4f}")