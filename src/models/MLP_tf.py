import tensorflow as tf
def train(X_train, y_train, X_test, y_test):
    model = tf.keras.Sequential([  
        # First hidden layer with 10 neurons, ReLU activation, and input shape of (4,) (4 features)
        tf.keras.layers.Dense(10, input_shape=(4,), activation='relu'),
        
        # Second hidden layer with 8 neurons and ReLU activation
        tf.keras.layers.Dense(8, activation='relu'),
        
        # Output layer with 3 neurons (for 3 classes) and softmax activation for classification
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=230, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f'Accuracy: {accuracy:.4f}')
