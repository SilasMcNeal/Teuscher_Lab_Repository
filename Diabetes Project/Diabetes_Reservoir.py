import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set verbosity and seed for reproducibility
rpy.verbosity(0)
rpy.set_seed(42)

# --- 1. Load Data ---
csv_file_path = "diabetes.csv"

if not os.path.exists(csv_file_path):
    print("Error: Could not find data file, please check file location.")
    exit()
else:
    print("Loading data")

try:
    df = pd.read_csv(csv_file_path)
except Exception as e:
    print(f"Error reading data: {e}")
    exit()

# --- 2. Preprocess Data ---
# Identify rows with missing 'Glucose' or 'Insulin' values
missing_condition = (df['Glucose'] == 0) | (df['Insulin'] == 0)

# Create a new target column 'Outcome_Class'
# 2 = "Could not decide" for unreliable data
df['Outcome_Class'] = df['Outcome']
df.loc[missing_condition, 'Outcome_Class'] = 2

# Define features (X) and target (y)
X = df.drop(columns=['Outcome', 'Outcome_Class'])
y = df['Outcome_Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

# --- 3. Build and Train ESN Model ---
# Reservoir parameters
units = 350
input_scaling = 1.0
lr = 0.8
sr = 0.01
regularization = 1e-5

# Define Reservoir and Readout nodes
reservoir = Reservoir(units, input_scaling=input_scaling, sr=sr, lr=lr)
readout = Ridge(ridge=regularization)

# Create the Echo State Network (ESN)
esn = reservoir >> readout

# Train the ESN
esn = esn.fit(x_train_scaled, y_train.values.reshape(-1, 1))

# --- 4. Evaluate Model ---
# Make predictions on the test set
prediction = esn.run(x_test_scaled)

# Process predictions
clipped_predictions = np.clip(prediction, 0, 2)
final_predictions = np.round(clipped_predictions).astype(int).flatten()

# Calculate and print overall accuracy
accuracy = accuracy_score(y_test, final_predictions)
print(f"Overall Model Accuracy: {accuracy:.2%}\n")

# Print the detailed classification report
class_labels = ['No Diabetes (0)', 'Diabetes (1)', 'Could Not Decide (2)']
report = classification_report(y_test, final_predictions, target_names=class_labels, zero_division=0)
print("Classification Report:")
print(report)

# Print a side-by-side comparison of actual vs. predicted
print("\nSample of Predictions vs. Actual Labels:")
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': final_predictions})
print(comparison_df.head(10))

# --- 5. Generate and Save Confusion Matrix ---
# Create the confusion matrix
conf_matrix = confusion_matrix(y_test, final_predictions)

# Plot the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

# Save the plot to a file
plt.savefig('confusion_matrix.png')

print("\nConfusion matrix saved to confusion_matrix.png")
