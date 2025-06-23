import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
# >>> Please update these three variables for your setup <<<
# 1. The full path to your CSV file.
CSV_FILE_PATH = "/u/silasm_guest/Tasks/Task_3/A_Z Handwritten Data.csv"

# 2. The height/width of your images (e.g., 28 for 28x28).
IMAGE_SIZE        = 28              

# 3. The exact name of the column in your CSV that contains the image labels.
LABEL_COLUMN_NAME = "0"        
# ---------------------

# --- Step 1: Load and Prepare Data from Your CSV ---
print("--- Data Loading & Preparation ---")
if not os.path.exists(CSV_FILE_PATH):
    print(f"FATAL ERROR: The file '{CSV_FILE_PATH}' was not found.")
    print("Please make sure the CSV_FILE_PATH variable is correct.")
    exit()

try:
    # Load the entire CSV file into a pandas DataFrame
    df = pd.read_csv(CSV_FILE_PATH)
except Exception as e:
    print(f"FATAL ERROR: Could not read the CSV file. Error: {e}")
    exit()

# Separate labels from pixel data
try:
    # Convert character/numeric labels into a 0-indexed integer format.
    df['category'] = df[LABEL_COLUMN_NAME].astype('category')
    labels_numeric = torch.tensor(df['category'].cat.codes.values, dtype=torch.long)
    # Create a dictionary to map the numeric codes back to original labels for visualization later.
    label_map = dict(enumerate(df['category'].cat.categories))
    # Loads pixel data into a pandas DataFrame. Creates a new DataFrame containing ONLY the pixel columns.
    pixels_df = df.drop([LABEL_COLUMN_NAME, 'category'], axis=1)
except KeyError:
    print(f"FATAL ERROR: The label column '{LABEL_COLUMN_NAME}' was not found.")
    exit()

# Data validation and conversion
num_expected_pixels = IMAGE_SIZE * IMAGE_SIZE
if num_expected_pixels != len(pixels_df.columns):
    print("FATAL ERROR: Image size mismatch!")
    exit()

#Converts DataFrame to a NumPy array(shape: [num_images, num_pixels]),
# torch.tensor converts array into pytorch tensor, /255.0 scales values to a range of [0,1]
pixels_tensor = torch.tensor(pixels_df.values, dtype=torch.float32) / 255.0
num_classes = len(label_map)
print(f"Successfully loaded {len(df)} samples with {num_classes} unique classes.")

#Reshapes the data into shape: (N, 1, 28, 28) (N samples, 1 channel, height, Width)
pixels_tensor_reshaped = pixels_tensor.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
#
dataset = TensorDataset(pixels_tensor_reshaped, labels_numeric)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --- Step 2: Define the PyTorch CNN Model ---
class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (IMAGE_SIZE//4) * (IMAGE_SIZE//4), 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CNNClassifier(num_classes=num_classes)
print("\n--- Using CNN Model Architecture ---")
print(model)

# --- Step 3: Define Loss, Optimizer and Start Training ---
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 15

# --- NEW: Variables to store history and track the best model ---
history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
best_accuracy = 0.0

print("\n--- Starting Training ---")
for epoch in range(num_epochs):
    # --- Training Phase ---
    model.train()
    running_train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
    
    # --- Validation Phase ---
    model.eval()
    running_val_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate average losses and accuracy for the epoch
    epoch_train_loss = running_train_loss / len(train_loader)
    epoch_val_loss = running_val_loss / len(val_loader)
    accuracy = 100 * correct / total

    # --- NEW: Record history ---
    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(accuracy)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Accuracy: {accuracy:.2f}%")

    # --- NEW: Check if this is the best model so far and save it ---
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"    -> New best model saved with accuracy: {accuracy:.2f}%")

print("\n--- Training Finished! ---")
print(f"Best validation accuracy achieved: {best_accuracy:.2f}%")

# --- Step 4: Plot Training History ---
print("\n--- Plotting Training History ---")
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy (%)')
plt.xlabel('Epoch')
plt.legend(['Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history['train_loss'])
plt.plot(history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot Training History
plt.tight_layout()
plt.savefig('training_history.png') # Save the plot to a file
plt.close() # Close the plot to free up memory
print("Saved training history plot to training_history.png")

# --- Step 5: Visualize Predictions on Validation Data ---
print("\n--- Visualizing Sample Predictions ---")
# Load the best model we saved for visualization
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

dataiter = iter(val_loader)
try:
    images, labels = next(dataiter)
except StopIteration:
    print("Validation loader is empty, cannot visualize.")
    exit()

# Get model predictions
with torch.no_grad():
    outputs = model(images)
_, predicted_numeric = torch.max(outputs, 1)

# Map predictions and true labels to original label format
predicted_labels = [label_map[p.item()] for p in predicted_numeric]
true_labels = [label_map[l.item()] for l in labels]

# Plot predictions with both true and predicted labels
plt.figure(figsize=(12, 6))
for i in range(min(10, len(images))):  # Plot up to 10 images
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i].squeeze(), cmap='gray')
    plt.title(f'True: {true_labels[i]}\nPred: {predicted_labels[i]}', fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig('prediction_examples_labeled.png')  # Save the plot
plt.close()
print("Saved prediction examples with labels to prediction_examples_labeled.png")
