from src.models.classifiers import Classifiers
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel
from sklearn.model_selection import train_test_split
import numpy as np

class Trainer:
    def __init__(self):
        pass
    
    def train_classifier(self, X, y, model_name='svm', test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        clf = Classifiers()
        model = clf.train(X_train, y_train, model_name)
        
        y_pred = clf.predict(X_test, model, model_name)
        accuracy = np.mean(y_pred == y_test)
        
        return model, accuracy
    
    def train_lstm(self, X, y, test_size=0.2, epochs=50, batch_size=32):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = len(np.unique(y))
        
        model = LSTMModel(input_shape, num_classes)
        history = model.train(
            X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size
        )
        
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_pred == y_test)
        
        return model, accuracy
    
    def train_transformer(self, X, y, test_size=0.2, epochs=50, batch_size=32):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = len(np.unique(y))
        
        model = TransformerModel(input_shape, num_classes)
        history = model.train(
            X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size
        )
        
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_pred == y_test)
        
        return model, accuracy