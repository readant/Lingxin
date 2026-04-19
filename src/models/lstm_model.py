import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LSTMModel:
    def __init__(self, input_shape, num_classes):
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        if X_val is not None and y_val is not None:
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size
            )
        return history
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model = tf.keras.models.load_model(path)