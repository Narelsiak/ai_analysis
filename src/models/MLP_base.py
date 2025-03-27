import numpy as np

def train(X_train, y_train, X_test, y_test):
    def relu(x):
        """ReLU activation function: returns max(0, x)"""
        return np.maximum(0, x)


    def relu_derivative(x):
        """Derivative of the ReLU function: 1 for x > 0, else 0"""
        return (x > 0).astype(float)


    def softmax(x):
        """Softmax activation function: normalizes logits into probabilities"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Prevent overflow
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


    def softmax_derivative(output, y):
        """Gradient of softmax with respect to the true labels"""
        return output - y

    # Define neural network architecture
    input_size = 4           # Number of input features
    hidden_size_1 = 10       # Number of neurons in the first hidden layer
    hidden_size_2 = 8        # Number of neurons in the second hidden layer
    output_size = 3          # Number of output classes

    # Initialize weights and biases
    np.random.seed(42)  # Seed for reproducibility
    W1 = np.random.randn(input_size, hidden_size_1) * 0.1
    b1 = np.zeros((1, hidden_size_1))
    W2 = np.random.randn(hidden_size_1, hidden_size_2) * 0.1
    b2 = np.zeros((1, hidden_size_2))
    W3 = np.random.randn(hidden_size_2, output_size) * 0.1
    b3 = np.zeros((1, output_size))

    # Training parameters
    learning_rate = 0.001
    epochs = 230

    def compute_loss(y_pred, y_true):
        """Computes categorical cross-entropy loss"""
        m = y_true.shape[0]  # Number of samples
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / m  # Avoid log(0)

    # Training loop
    for epoch in range(epochs):
        """Forward pass"""
        Z1 = np.dot(X_train, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = relu(Z2)
        Z3 = np.dot(A2, W3) + b3
        A3 = softmax(Z3)

        # Compute loss
        loss = compute_loss(A3, y_train)

        """Backward pass (Gradient computation)"""
        dZ3 = softmax_derivative(A3, y_train)
        dW3 = np.dot(A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dZ2 = np.dot(dZ3, W3.T) * relu_derivative(Z2)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dZ1 = np.dot(dZ2, W2.T) * relu_derivative(Z1)
        dW1 = np.dot(X_train.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        """Gradient descent parameter update"""
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3

    # Evaluate on test set
    Z1_test = np.dot(X_test, W1) + b1
    A1_test = relu(Z1_test)
    Z2_test = np.dot(A1_test, W2) + b2
    A2_test = relu(Z2_test)
    Z3_test = np.dot(A2_test, W3) + b3
    A3_test = softmax(Z3_test)

    # Compute accuracy
    predictions = np.argmax(A3_test, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == y_test_classes)
    print(f"Test Accuracy: {accuracy:.4f}")