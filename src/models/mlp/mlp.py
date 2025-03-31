import numpy as np
from .optimizers import SGD

class MLP:
    def __init__(self, layers, optimizer=SGD()):
        self.layers = layers
        self.optimizer = optimizer

        for i in range(0, len(self.layers)):
            if self.layers[i].input_size is None:
                self.layers[i].input_size = self.layers[i - 1].output_size
                self.layers[i]._init_weights()

            if self.optimizer:
                self.layers[i].optimizer = self.optimizer
    
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X 
    
    def backward(self, dA):
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
        return dA
    
    def fit(self, X_train, y_train, epochs=100, batch_size=32):
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            self.optimizer.update_lr(epoch)

            indices = np.random.permutation(n_samples)
            X_train, y_train = X_train[indices], y_train[indices]

            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                output = self.forward(X_batch)
                
                loss = -np.sum(y_batch * np.log(output + 1e-8)) / y_batch.shape[0]

                dA = output - y_batch

                self.backward(dA)
                
                epoch_loss += loss

            loss = epoch_loss / (X_train.shape[0] // batch_size)
            accuracy = np.mean(np.argmax(output, axis=1) == np.argmax(y_batch, axis=1))
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    def predict(self, X, y):
        output = self.forward(X)
        predictions = np.argmax(output, axis=1)
        y_true = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y_true)
        return predictions, accuracy

    
    def summary(self):
        print("Network Summary:")
        print("=" * 40)
        print(f"{'Layer':<10}{'Input':<10}{'Output':<10}{'Activation':<10}")
        print("-" * 40)
        for i, layer in enumerate(self.layers):
            activation_name = layer.activation.__name__ if callable(layer.activation) else layer.activation
            print(f"{i+1:<10}{layer.input_size:<10}{layer.output_size:<10}{activation_name:<10}")
        print("=" * 40)