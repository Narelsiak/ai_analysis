import tensorflow as tf
import numpy as np

def train(X_train, y_train, X_test, y_test, epochs=230, learning_rate=0.01):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(4,), activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=epochs, verbose=0, validation_data=(X_test, y_test))
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    predictions = np.argmax(model.predict(X_test), axis=1)
    y_test_classes = np.argmax(y_test, axis=1)    
    
    return history.history['loss'], history.history['accuracy'], predictions, y_test_classes, accuracy