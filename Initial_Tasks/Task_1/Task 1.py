import numpy as np
from matplotlib import pyplot as plt

# --- 1. Setup the Data ---
# XOR inputs and corresponding labels
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# --- 2. Define Architecture and Initialize Weights ---
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

hidden_weights = np.random.rand(input_neurons, hidden_neurons)
hidden_bias = np.random.rand(1, hidden_neurons)
output_weights = np.random.rand(hidden_neurons, output_neurons)
output_bias = np.random.rand(1, output_neurons)


# --- Define Activation Function and its Derivative ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# --- Training the Network ---
learning_rate = 0.7
epochs = 900
loss_history = []

print("Training started...")
for i in range(epochs):
    # --- Forward Propagation ---
    hidden_layer_input = np.dot(X, hidden_weights) + hidden_bias
    hidden_layer_activation = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_activation, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_input)

    # --- Backpropagation ---
    error = y - predicted_output

    # This is the corrected line
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    # The rest of the backpropagation logic
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activation)

    # --- Update Weights and Biases ---
    output_weights += hidden_layer_activation.T.dot(d_predicted_output) * learning_rate
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    hidden_weights += X.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    loss_history.append(np.mean(np.square(error)))
    loss = np.mean(np.square(error))
    print(f"Epoch {i}, Loss: {loss:.4f}")


print("\nTraining finished!")

# --- Making Predictions ---
print("\nPredictions after training:")
print("Input | Raw Output   | Rounded Prediction")
print("------------------------------------------")
raw_outputs = sigmoid(sigmoid(np.dot(X, hidden_weights) + hidden_bias).dot(output_weights) + output_bias)
for i in range(len(X)):
    raw_output = raw_outputs[i][0]
    rounded = 1 if raw_output > 0.5 else 0
    print(f"{X[i]}  | {raw_output:.8f} | {rounded}")

x = np.linspace(0, epochs, epochs)
y = loss_history

plt.plot(x, loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
