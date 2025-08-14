import torch
import torch.nn as nn
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

# --- Data preparation ---
csv_file_path = "glucose.csv" # Path to csv file

#ensures the path is correct
if not os.path.exists(csv_file_path):
    print(f"Could not load data, please check that {csv_file_path} exists.")
else:
    print(f"Loading data and starting the Network..")

# Load the CSV file
try:
    df = pd.read_csv(csv_file_path)
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    exit()

#prep our data
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df.sort_values(by='datetime', inplace=True)
df_cgm = df[df['type'] == 'cgm'].copy()
df_cgm['glucose'] = pd.to_numeric(df_cgm['glucose']) #makes sure our glucose values are numbers
df_cgm.set_index('datetime', inplace=True) #setting datetime as index

df_cgm = df_cgm[['glucose']].resample('5T').mean().interpolate(method='linear') #resamples to 5 min intervals and fills missing values

# --- Feature engineering ---
df_cgm['glucose_in_30'] = df_cgm['glucose'].shift(-6)  # 30 minutes ahead
df_cgm['glucose_change'] = df_cgm['glucose_in_30'] - df_cgm['glucose'] #the change in glucose
df_cgm["is_major_drop"] = (df_cgm['glucose_change'] <= -2.0).astype(int)  # Major drop if glucose drops by more than 2.0 mmol/L
df_cgm.dropna(subset=['glucose_in_30', 'glucose_change', 'is_major_drop'], inplace=True) #removes rows with missing values
print(df_cgm['is_major_drop'].value_counts(normalize=True))

window_size = 12
glucose_values = df_cgm['glucose'].values
labels = df_cgm['is_major_drop'].values

#declare our numpy arrays
X = []
y = []

for i in range(len(df_cgm) - window_size):
    X.append(glucose_values[i:i+window_size]) #selects a window (6) glucose readings
    y.append(labels[i + window_size])

#convert arrays to numpy arrays
X = np.array(X)
y = np.array(y)

#Splits the data into training and test sets 80/20,
# stratified by the label to keep balance. 
# random seed is set to 50 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50, stratify=y) 

# 1. Initialize the Scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# 2. Reshape data for scaling
n_samples_train, window_size = X_train.shape
n_samples_test, _ = X_test.shape
X_train_reshaped = X_train.reshape(-1, 1)
X_test_reshaped = X_test.reshape(-1, 1)

# 3. Fit the scaler on the TRAINING data and transform it
print("Fitting the scaler on training data and transforming...")
X_train_scaled_reshaped = scaler.fit_transform(X_train_reshaped)

# 4. Transform the TEST data using the *same* fitted scaler
print("Transforming test data with the fitted scaler...")
X_test_scaled_reshaped = scaler.transform(X_test_reshaped)

# 5. Reshape the data back to its original shape for the LSTM
X_train_scaled = X_train_scaled_reshaped.reshape(n_samples_train, window_size)
X_test_scaled = X_test_scaled_reshaped.reshape(n_samples_test, window_size)

#wraps data for pytorch, converts to tensors
class GlucoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
#dataset instances
train_dataset = GlucoseDataset(X_train_scaled, y_train)
test_dataset = GlucoseDataset(X_test_scaled, y_test)

#dataloader instances, batch size
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#--- Neural Network Model ---
class GlucoseNetwork(nn.Module):
    #network parameters
    hidden_neurons = 64
    num_layers = 2
    input_neurons = 1
    output_neurons = 1

    def __init__(self, input_neurons, hidden_neurons, num_layers, output_neurons):
        super(GlucoseNetwork, self).__init__()
        self.hidden_neurons = hidden_neurons
        self.num_layers = num_layers

        #LSTM Layer
        self.LSTM = nn.LSTM(input_neurons, hidden_neurons, num_layers, batch_first=True, dropout=0.2)

        #fully connected layer
        self.fc = nn.Linear(hidden_neurons, output_neurons)

    def forward(self, x):
         # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_neurons).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_neurons).to(x.device)

        # We pass the input and the initial hidden/cell states to the LSTM
        out, _ = self.LSTM(x, (h0, c0))

        # We take the output of the last time step
        out = self.fc(out[:, -1, :])

        # No sigmoid here
        return out

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = GlucoseNetwork(
    input_neurons=1,
    hidden_neurons=64,
    num_layers=2,
    output_neurons=1
).to(device)
#takes input of (batch, sequence_length, input_size (1))

print("\nModel architecture:")
print(model)

#hyperparameters
num_epochs = 350
learning_rate = 0.0032945552201069143

# Calculate pos_weight for BCEWithLogitsLoss
num_zeros = np.sum(y_train == 0)
num_ones = np.sum(y_train == 1)
pos_weight = torch.tensor([num_zeros / num_ones], dtype=torch.float32).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Weighted loss for imbalance
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []
accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.unsqueeze(-1).to(device)
        batch_y = batch_y.to(device).unsqueeze(-1)

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(batch_X)  # Forward pass
        loss = criterion(outputs, batch_y)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item() * batch_X.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # Calculate accuracy on test set after each epoch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.unsqueeze(-1).to(device)
            batch_y = batch_y.to(device).unsqueeze(-1)
            outputs = model(batch_X)
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    accuracy = correct / total
    losses.append(epoch_loss)
    accuracies.append(accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

# Plot loss and accuracy curves after training
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(accuracies, label='Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')  # Save the figure to a PNG file
plt.close()
print('Training curves saved to training_curves.png')

# --- Generate and Plot Confusion Matrix ---
print("\nGenerating confusion matrix on the test set...")
model.eval() # Set the model to evaluation mode

all_labels = []
all_predictions = []

# No gradient is needed for evaluation
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        # Move data to the appropriate device
        batch_X = batch_X.unsqueeze(-1).to(device)

        # Get model outputs
        outputs = model(batch_X)

        # Apply sigmoid and threshold to get predictions (0 or 1)
        predicted = (torch.sigmoid(outputs) >= 0.5).float()

        # Move predictions and labels to CPU and convert to numpy for sklearn
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

# Convert lists of lists to a flat numpy array
all_labels = np.array(all_labels).flatten()
all_predictions = np.array(all_predictions).flatten()

# Compute the confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Drop', 'Major Drop'],
            yticklabels=['No Drop', 'Major Drop'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

# Save to file
plt.savefig('confusion_matrix.png')
plt.close()

print('Confusion matrix saved to confusion_matrix.png')
