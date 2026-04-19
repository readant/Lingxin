import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D

class TransformerModel:
    def __init__(self, input_shape, num_classes, d_model=64, num_heads=2, dff=128, dropout_rate=0.1):
        inputs = Input(shape=input_shape)
        x = inputs
        
        for _ in range(2):
            attn_output = MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model
            )(x, x)
            attn_output = Dropout(dropout_rate)(attn_output)
            out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)
            
            ffn_output = Dense(dff, activation='relu')(out1)
            ffn_output = Dense(d_model)(ffn_output)
            ffn_output = Dropout(dropout_rate)(ffn_output)
            x = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        
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