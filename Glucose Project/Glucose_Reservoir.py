import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from reservoirpy.nodes import Reservoir

# Helper context manager to suppress verbose output from libraries
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr.close()
        sys.stderr = self._original_stderr

def main():
    csv_file_path = "glucose.csv"

    if not os.path.exists(csv_file_path):
        print(f"Could not load data, please check that {csv_file_path} exists.")
        return
    else:
        print(f"Loading data and preparing for the Network...")

    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return
    
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.sort_values(by='datetime', inplace=True)
    df_cgm = df[df['type'] == 'cgm'].copy()
    df_cgm['glucose'] = pd.to_numeric(df_cgm['glucose'])
    df_cgm.set_index('datetime', inplace=True)
    df_cgm = df_cgm[['glucose']].resample('5min').mean().interpolate(method='linear')

    df_cgm['glucose_in_30'] = df_cgm['glucose'].shift(-6)  # 30 minutes ahead (6 * 5min intervals)
    df_cgm['glucose_change'] = df_cgm['glucose_in_30'] - df_cgm['glucose']
    df_cgm["is_major_drop"] = (df_cgm['glucose_change'] <= -2.0).astype(int)
    df_cgm.dropna(subset=['glucose_in_30', 'glucose_change', 'is_major_drop'], inplace=True)
    print("\nClass distribution ('1' is a major drop):")
    print(df_cgm['is_major_drop'].value_counts(normalize=True))

    window_size = 12  #Use 12 readings (1 hour) to predict
    glucose_values = df_cgm['glucose'].values
    labels = df_cgm['is_major_drop'].values

    X, y = [], []
    for i in range(len(df_cgm) - window_size):
        X.append(glucose_values[i:i+window_size])
        y.append(labels[i + window_size])

    X, y = np.array(X), np.array(y)
    
    # Reshape X to be (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50, stratify=y)

    #scaling data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_reshaped = X_train.reshape(-1, 1)
    X_test_reshaped = X_test.reshape(-1, 1)

    print("\nFitting the scaler on training data and transforming")
    scaler.fit(X_train_reshaped)
    X_train_scaled_reshaped = scaler.transform(X_train_reshaped)
    X_test_scaled_reshaped = scaler.transform(X_test_reshaped)

    X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)
    X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape)
    print(f"Data prepared: {X_train_scaled.shape[0]} training samples, {X_test_scaled.shape[0]} testing samples.")
    print("\nSetting up reservoir and training the readout")
    
    reservoir = Reservoir(units=500, lr=0.24, sr=1.0, name="reservoir")

    # Get the last reservoir state for each training sequence
    train_states = []
    for x_sample in tqdm(X_train_scaled, desc="Training Progress"):
        with SuppressOutput(): #keeps the terminal clear from messages from the .run function
            all_sample_states = reservoir.run(x_sample, reset=True) #resets the state to 0 before processing the next sample
        train_states.append(all_sample_states[-1])
    train_states = np.array(train_states)

    readout = LogisticRegression(max_iter=1000, class_weight='balanced')
    readout.fit(train_states, y_train)
    print("Training complete.")
    print("\nEvaluating the model on the test set...")
    
    #gets the last state in the reservoir
    test_states = []
    for x_sample in tqdm(X_test_scaled, desc="Testing Progress "):
        with SuppressOutput(): # Suppress verbose output from the run method
            all_sample_states = reservoir.run(x_sample, reset=True) # Reset state for each independent sample
        test_states.append(all_sample_states[-1])
    test_states = np.array(test_states)
    
    probabilities = readout.predict_proba(test_states)

   
    custom_threshold = 0.66
    #selects all rows in numpy arrary from the second column to apply threshold to
    predictions = (probabilities[:, 1] >= custom_threshold).astype(int)
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['No Drop', 'Major Drop']))
    
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Drop', 'Major Drop'], 
                yticklabels=['No Drop', 'Major Drop'])
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig("results/glucose_confusion_matrix.png")
    print("\nConfusion matrix plot saved as 'glucose_confusion_matrix.png'")
    plt.show()

if __name__ == "__main__":
    main()
