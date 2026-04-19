import torch
from torch.utils.data import DataLoader, TensorDataset
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
    
    def train_lstm(self, X, y, test_size=0.2, epochs=50, batch_size=32, lr=0.001):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = len(np.unique(y))
        
        model = LSTMModel(input_shape, num_classes)
        model.train_model(train_loader, val_loader, epochs=epochs, lr=lr)
        
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            _, y_pred = torch.max(y_pred, 1)
            accuracy = (y_pred == y_test).float().mean().item()
        
        return model, accuracy
    
    def train_transformer(self, X, y, test_size=0.2, epochs=50, batch_size=32, lr=0.001):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = len(np.unique(y))
        
        model = TransformerModel(input_shape, num_classes)
        model.train_model(train_loader, val_loader, epochs=epochs, lr=lr)
        
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            _, y_pred = torch.max(y_pred, 1)
            accuracy = (y_pred == y_test).float().mean().item()
        
        return model, accuracy