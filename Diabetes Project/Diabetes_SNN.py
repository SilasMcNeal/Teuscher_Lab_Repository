import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns

csv_file_path = "glucose.csv" 

if not os.path.exists(csv_file_path):
    print(f"Error: Could not find the data file at '{csv_file_path}'.")
    print("Please ensure the 'glucose.csv' file is in the correct directory before running.")
    exit()
else:
    print(f"Loading data from '{csv_file_path}' and starting the Network.")

try:
    df = pd.read_csv(csv_file_path)
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    exit()

df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df.sort_values(by='datetime', inplace=True)
df_cgm = df[df['type'] == 'cgm'].copy()
df_cgm['glucose'] = pd.to_numeric(df_cgm['glucose'])
df_cgm.set_index('datetime', inplace=True)

df_cgm = df_cgm[['glucose']].resample('5min').mean().interpolate(method='linear')

# --- Feature engineering ---
df_cgm['glucose_in_30'] = df_cgm['glucose'].shift(-6)
df_cgm['glucose_change'] = df_cgm['glucose_in_30'] - df_cgm['glucose']
df_cgm["is_major_drop"] = (df_cgm['glucose_change'] <= -2.0).astype(int)
df_cgm.dropna(subset=['glucose_in_30', 'glucose_change', 'is_major_drop'], inplace=True)
print("Class distribution:")
print(df_cgm['is_major_drop'].value_counts(normalize=True))

window_size = 12
glucose_values = df_cgm['glucose'].values
labels = df_cgm['is_major_drop'].values

X = []
y = []

for i in range(len(df_cgm) - window_size):
    X.append(glucose_values[i:i+window_size])
    y.append(labels[i + window_size])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50, stratify=y) 

# --- Scaling the data ---
scaler = MinMaxScaler(feature_range=(0, 1))

n_samples_train, window_size_shape = X_train.shape
n_samples_test, _ = X_test.shape
X_train_reshaped = X_train.reshape(-1, 1)
X_test_reshaped = X_test.reshape(-1, 1)

print("Fitting the scaler on training data and transforming...")
X_train_scaled_reshaped = scaler.fit_transform(X_train_reshaped)

print("Transforming test data with the fitted scaler...")
X_test_scaled_reshaped = scaler.transform(X_test_reshaped)

X_train_scaled = X_train_scaled_reshaped.reshape(n_samples_train, window_size_shape)
X_test_scaled = X_test_scaled_reshaped.reshape(n_samples_test, window_size_shape)

class GlucoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 32
train_dataset = GlucoseDataset(X_train_scaled, y_train)
test_dataset = GlucoseDataset(X_test_scaled, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class SNN_GlucoseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, beta=0.9):
        super().__init__()
        
        # Use the atan surrogate gradient function for smoother learning.
        spike_grad = surrogate.fast_sigmoid(slope = 15) #surrogate.atan()
        
        # Layer 1
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.lif1 = snn.Leaky(beta=beta, learn_beta=True, spike_grad=spike_grad, reset_mechanism="zero")

        # Layer 2
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.lif2 = snn.Leaky(beta=beta, learn_beta=True, spike_grad=spike_grad, reset_mechanism="zero")

        # Layer 3 (Output layer)
        self.fc3 = nn.Linear(hidden_size_2, output_size)
        self.lif3 = snn.Leaky(beta=beta, learn_beta=True, spike_grad=spike_grad, reset_mechanism="zero")

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        mem3_recorder = []

        # Permute data to [time, batch, features]
        x_seq = x.permute(1, 0, 2)

        # Process data over time steps
        for step in range(x_seq.size(0)): 
            cur1 = self.fc1(x_seq[step])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            mem3_recorder.append(mem3)
        
        recorded_mem = torch.stack(mem3_recorder, dim=0)

        # Average the membrane potential over all time steps
        final_mem_mean = torch.mean(recorded_mem, dim=0)
        
        return final_mem_mean

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

input_neurons_snn = 1 

model = SNN_GlucoseNetwork(
    input_size=input_neurons_snn,
    hidden_size_1=64,
    hidden_size_2=64, 
    output_size=1
).to(device)

print("\nSNN Model architecture:")
print(model)

# --- Training Setup ---
num_epochs = 500
learning_rate = 1e-3

# Calculate class weights to handle imbalance
num_zeros = np.sum(y_train == 0)
num_ones = np.sum(y_train == 1)

if num_ones > 0:
    pos_weight_val = num_zeros / num_ones
else:
    pos_weight_val = 1.0 
    
pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []
accuracies = []

print("\n--- Starting Training ---")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (data, targets) in enumerate(train_loader):
        data = data.unsqueeze(-1).to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        
        final_output = model(data)
        
        loss = criterion(final_output, targets)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    # --- Evaluation on Test Set ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.unsqueeze(-1).to(device)
            batch_y = batch_y.to(device).unsqueeze(-1)
            
            final_output = model(batch_X)
            
            predicted = (torch.sigmoid(final_output) >= 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            
    accuracy = correct / total
    losses.append(epoch_loss)
    accuracies.append(accuracy)
    
  
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Test Accuracy: {accuracy:.4f}")

# --- Plotting Results ---
print("\n--- Training Finished ---")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot training loss
ax1.plot(losses, label='Training Loss', color='blue')
ax1.set_title('Training Loss vs. Epochs')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss') 
ax1.grid(True)
ax1.legend()

# Plot test accuracy
ax2.plot(accuracies, label='Test Accuracy', color='green')
ax2.set_title('Test Accuracy vs. Epochs')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.grid(True)
ax2.legend()

# Display and save the plots
plt.tight_layout()
plt.savefig("output.png")
plt.show()

# --- Confusion Matrix ---
print("\n--- Generating Confusion Matrix on Final Model ---")
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.unsqueeze(-1).to(device)

        # Get model predictions
        final_output = model(batch_X)
        predicted = (torch.sigmoid(final_output) >= 0.5).float()

        # Append predictions and true labels to lists
        all_preds.extend(predicted.cpu().numpy().flatten())
        all_targets.extend(batch_y.cpu().numpy().flatten())

# Create and save the confusion matrix
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Major Drop', 'Major Drop'],
            yticklabels=['No Major Drop', 'Major Drop'])
plt.title('Confusion Matrix on Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("confusion_matrix.png")
plt.show()

print("\nConfusion matrix saved to 'confusion_matrix.png'")
