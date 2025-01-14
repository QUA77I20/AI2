import numpy as np 

# Sigmoid function to transform input values to a range between 0 and 1.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Training dataset with binary input values.
training_inputs = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
])

# Target outputs for the training dataset (correct results expected from the network).
training_outputs = np.array([[0, 1, 1, 0]]).T

# Setting a random seed to ensure reproducibility of the results.
np.random.seed(1)

# Initializing synaptic weights randomly with values between -1 and 1.
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Initial random weights:")
print(synaptic_weights)

# Training the neural network using the backpropagation method.
for i in range(500000):
    # Input layer (taking the training inputs).
    input_layer = training_inputs

    # Calculating the outputs by applying the sigmoid function to the dot product of inputs and weights.
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    # Calculating the error by subtracting the predicted outputs from the target outputs.
    error = training_outputs - outputs

    # Adjusting the weights using the backpropagation algorithm.
    adjustments = np.dot(input_layer.T, error * (outputs * (1 - outputs)))

    # Updating the synaptic weights.
    synaptic_weights += adjustments

print("Weights after training:")
print(synaptic_weights)

print("Output after training:")
print(outputs)

# Testing the trained neural network with new input data.
new_inputs = np.array([1, 1, 0])
output = sigmoid(np.dot(new_inputs, synaptic_weights))
print("Test output:")
print(output)