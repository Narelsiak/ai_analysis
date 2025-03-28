import numpy as np

def train(X_train, y_train, X_test, y_test, epochs=230, learning_rate=0.01):
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

    # Initialize weights and biases
    W1 = np.random.randn(input_size, hidden_size_1) * 0.1
    b1 = np.zeros((1, hidden_size_1))
    W2 = np.random.randn(hidden_size_1, hidden_size_2) * 0.1
    b2 = np.zeros((1, hidden_size_2))
    W3 = np.random.randn(hidden_size_2, output_size) * 0.1
    b3 = np.zeros((1, output_size))

    lambda_reg = 0.01  # L2 regularization strength

    def compute_loss(y_pred, y_true):
        m = y_true.shape[0]
        # Cross-entropy loss with a small epsilon for numerical stability
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / m

    loss_history, accuracy_history = [], []

    for epoch in range(epochs):
        Z1 = np.dot(X_train, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = relu(Z2)
        Z3 = np.dot(A2, W3) + b3
        A3 = softmax(Z3)

        # Compute the loss, including L2 regularization term
        # L2 regularization penalizes large weights to prevent overfitting
        loss = compute_loss(A3, y_train) + lambda_reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))

        predictions_train = np.argmax(A3, axis=1)
        y_train_classes = np.argmax(y_train, axis=1)
        accuracy_train = np.mean(predictions_train == y_train_classes)

        loss_history.append(loss)
        accuracy_history.append(accuracy_train)

        dZ3 = softmax_derivative(A3, y_train)
        dW3 = np.dot(A2.T, dZ3) + lambda_reg * W3  # Add the regularization term to the gradient
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dZ2 = np.dot(dZ3, W3.T) * relu_derivative(Z2)
        dW2 = np.dot(A1.T, dZ2) + lambda_reg * W2  # Add the regularization term to the gradient
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dZ1 = np.dot(dZ2, W2.T) * relu_derivative(Z1)
        dW1 = np.dot(X_train.T, dZ1) + lambda_reg * W1  # Add the regularization term to the gradient
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Gradient Descent updates
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3

    Z1_test = np.dot(X_test, W1) + b1
    A1_test = relu(Z1_test)
    Z2_test = np.dot(A1_test, W2) + b2
    A2_test = relu(Z2_test)
    Z3_test = np.dot(A2_test, W3) + b3
    A3_test = softmax(Z3_test)

    predictions = np.argmax(A3_test, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == y_test_classes)    

    return loss_history, accuracy_history, y_test_classes, predictions, accuracy
