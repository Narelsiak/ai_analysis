class MLP:
    def __init__(self, layers):
        self.layers = layers
        
        for i in range(1, len(self.layers)):
            if self.layers[i].input_size is None:
                self.layers[i].input_size = self.layers[i - 1].output_size
                self.layers[i]._init_weights()
    
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, dA, learning_rate):
        for layer in reversed(self.layers):
            dA = layer.backward(dA, learning_rate)
        return dA
    
    def fit(self, X_train, y_train, epochs=100, learning_rate=0.01, optimizer=None):
        # batch_size = 32
        for epoch in range(epochs):
            output = self.forward(X_train)

            #loss = 
            pass

    def summary(self):
        print("Network Summary:")
        print("=" * 40)
        print(f"{'Layer':<10}{'Input':<10}{'Output':<10}{'Activation':<10}")
        print("-" * 40)
        for i, layer in enumerate(self.layers):
            activation_name = layer.activation.__name__ if callable(layer.activation) else layer.activation
            print(f"{i+1:<10}{layer.input_size:<10}{layer.output_size:<10}{activation_name:<10}")
        print("=" * 40)