import numpy as np

def train(X_train, y_train, X_test, y_test):
    def relu(x):
        return np.maximum(0, x)

    def relu_derivative(x):
        return (x > 0).astype(float)

    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def softmax_derivative(output, y):
        return output - y

    input_size = 4
    hidden_size_1 = 10
    hidden_size_2 = 8
    output_size = 3

    np.random.seed(42)

    # Initialize weights and biases with small random values
    W1 = np.random.randn(input_size, hidden_size_1) * 0.1
    b1 = np.zeros((1, hidden_size_1))
    W2 = np.random.randn(hidden_size_1, hidden_size_2) * 0.1
    b2 = np.zeros((1, hidden_size_2))
    W3 = np.random.randn(hidden_size_2, output_size) * 0.1
    b3 = np.zeros((1, output_size))

    learning_rate = 0.01
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8  # Hyperparameters for Adam
    epochs = 230

    # Initialize moment estimates for Adam (first moment, second moment)
    mW1, vW1, mb1, vb1 = 0, 0, 0, 0
    mW2, vW2, mb2, vb2 = 0, 0, 0, 0
    mW3, vW3, mb3, vb3 = 0, 0, 0, 0

    def compute_loss(y_pred, y_true):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / m

    for epoch in range(epochs):
        # Forward pass: compute activations
        Z1 = np.dot(X_train, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = relu(Z2)
        Z3 = np.dot(A2, W3) + b3
        A3 = softmax(Z3)

        # Compute loss
        loss = compute_loss(A3, y_train)

        # Backpropagation: compute gradients for each layer
        dZ3 = softmax_derivative(A3, y_train)
        dW3 = np.dot(A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dZ2 = np.dot(dZ3, W3.T) * relu_derivative(Z2)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dZ1 = np.dot(dZ2, W2.T) * relu_derivative(Z1)
        dW1 = np.dot(X_train.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Adam optimizer updates
        for param, dparam, m, v in zip([W1, W2, W3, b1, b2, b3],
                                       [dW1, dW2, dW3, db1, db2, db3],
                                       [mW1, mW2, mW3, mb1, mb2, mb3],
                                       [vW1, vW2, vW3, vb1, vb2, vb3]):
            # Update biased first moment estimate (m)
            m = beta1 * m + (1 - beta1) * dparam
            # Update biased second moment estimate (v)
            v = beta2 * v + (1 - beta2) * (dparam ** 2)
            
            # Compute bias-corrected first moment estimate (m_hat)
            m_hat = m / (1 - beta1 ** (epoch + 1))
            # Compute bias-corrected second moment estimate (v_hat)
            v_hat = v / (1 - beta2 ** (epoch + 1))
            
            # Update the parameters using the Adam optimization rule
            param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    # Forward pass on the test set
    Z1_test = np.dot(X_test, W1) + b1
    A1_test = relu(Z1_test)
    Z2_test = np.dot(A1_test, W2) + b2
    A2_test = relu(Z2_test)
    Z3_test = np.dot(A2_test, W3) + b3
    A3_test = softmax(Z3_test)

    # Compute predictions and accuracy
    predictions = np.argmax(A3_test, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == y_test_classes)
    
    print(f"Test Accuracy: {accuracy:.4f}")
