import numpy as np

def train(X_train, y_train, X_test, y_test, epochs=230, learning_rate=0.01, batch_size=32, lr_scheduler=True, step_size=10, gamma=0.95):
    def relu(x):
        return np.maximum(0, x)

    def relu_derivative(x):
        return (x > 0).astype(float)

    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def softmax_derivative(output, y):
        return output - y

    input_size = X_train.shape[1]
    hidden_size_1 = 10
    hidden_size_2 = 8
    output_size = y_train.shape[1]

    np.random.seed(42)

    # Initialize weights using He initialization
    W1 = np.random.randn(input_size, hidden_size_1) * np.sqrt(2 / input_size)
    W2 = np.random.randn(hidden_size_1, hidden_size_2) * np.sqrt(2 / hidden_size_1)
    W3 = np.random.randn(hidden_size_2, output_size) * np.sqrt(2 / hidden_size_2)

    # Biases initialized to zero
    b1 = np.zeros((1, hidden_size_1))
    b2 = np.zeros((1, hidden_size_2))
    b3 = np.zeros((1, output_size))

    beta1, beta2, epsilon = 0.9, 0.999, 1e-8  # Hyperparameters for Adam

    # Initialize moment estimates for Adam (first moment, second moment)
    mW1, vW1, mb1, vb1 = np.zeros_like(W1), np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(b1)
    mW2, vW2, mb2, vb2 = np.zeros_like(W2), np.zeros_like(W2), np.zeros_like(b2), np.zeros_like(b2)
    mW3, vW3, mb3, vb3 = np.zeros_like(W3), np.zeros_like(W3), np.zeros_like(b3), np.zeros_like(b3)

    def compute_loss(y_pred, y_true):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / m

    loss_history, accuracy_history = [], []

    for epoch in range(epochs):
        # Shuffle data before each epoch
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train, y_train = X_train[indices], y_train[indices]

        epoch_loss = 0
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Forward pass
            Z1 = np.dot(X_batch, W1) + b1
            A1 = relu(Z1)
            Z2 = np.dot(A1, W2) + b2
            A2 = relu(Z2)
            Z3 = np.dot(A2, W3) + b3
            A3 = softmax(Z3)

            # Compute loss
            loss = compute_loss(A3, y_batch)

            # Backpropagation
            dZ3 = softmax_derivative(A3, y_batch)
            dW3 = np.dot(A2.T, dZ3)
            db3 = np.sum(dZ3, axis=0, keepdims=True)

            dZ2 = np.dot(dZ3, W3.T) * relu_derivative(Z2)
            dW2 = np.dot(A1.T, dZ2)
            db2 = np.sum(dZ2, axis=0, keepdims=True)

            dZ1 = np.dot(dZ2, W2.T) * relu_derivative(Z1)
            dW1 = np.dot(X_batch.T, dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)

            # Adam optimizer updates
            for param, dparam, m, v in zip([W1, W2, W3, b1, b2, b3],
                                           [dW1, dW2, dW3, db1, db2, db3],
                                           [mW1, mW2, mW3, mb1, mb2, mb3],
                                           [vW1, vW2, vW3, vb1, vb2, vb3]):
                m = beta1 * m + (1 - beta1) * dparam
                v = beta2 * v + (1 - beta2) * (dparam ** 2)
                m_hat = m / (1 - beta1 ** (epoch + 1))
                v_hat = v / (1 - beta2 ** (epoch + 1))
                param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            epoch_loss += loss

        # Update learning rate with scheduler
        if lr_scheduler and epoch % step_size == 0 and epoch != 0:
            learning_rate *= gamma
            #print(learning_rate)
        # Compute training accuracy at the end of the epoch
        Z1_train = np.dot(X_train, W1) + b1
        A1_train = relu(Z1_train)
        Z2_train = np.dot(A1_train, W2) + b2
        A2_train = relu(Z2_train)
        Z3_train = np.dot(A2_train, W3) + b3
        A3_train = softmax(Z3_train)

        predictions_train = np.argmax(A3_train, axis=1)
        y_train_classes = np.argmax(y_train, axis=1)
        accuracy_train = np.mean(predictions_train == y_train_classes)

        loss_history.append(epoch_loss / (X_train.shape[0] // batch_size))
        accuracy_history.append(accuracy_train)

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
        
    return loss_history, accuracy_history, predictions, y_test_classes, accuracy
