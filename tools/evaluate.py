import numpy as np
import os
from src.utils.metrics import Metrics
from src.models.classifiers import Classifiers
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel

class EvaluateRunner:
    def __init__(self):
        self.metrics = Metrics()
    
    def run(self, data_dir, model_type='svm', model_path=None):
        if model_type in ['svm', 'rf', 'mlp']:
            X = np.load(os.path.join(data_dir, 'X.npy'))
            y = np.load(os.path.join(data_dir, 'y.npy'))
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            clf = Classifiers()
            model = clf.train(X_train, y_train, model_type)
            y_pred = clf.predict(X_test, model, model_type)
        elif model_type == 'lstm':
            X = np.load(os.path.join(data_dir, 'X_sequence.npy'))
            y = np.load(os.path.join(data_dir, 'y_sequence.npy'))
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            model = LSTMModel((X.shape[1], X.shape[2]), len(np.unique(y)))
            model.train(X_train, y_train, X_test, y_test, epochs=10)
            y_pred = model.predict(X_test)
            y_pred = np.argmax(y_pred, axis=1)
        elif model_type == 'transformer':
            X = np.load(os.path.join(data_dir, 'X_sequence.npy'))
            y = np.load(os.path.join(data_dir, 'y_sequence.npy'))
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            model = TransformerModel((X.shape[1], X.shape[2]), len(np.unique(y)))
            model.train(X_train, y_train, X_test, y_test, epochs=10)
            y_pred = model.predict(X_test)
            y_pred = np.argmax(y_pred, axis=1)
        else:
            print(f'Unknown model type: {model_type}')
            return
        
        metrics = self.metrics.calculate_metrics(y_test, y_pred)
        print('Evaluation Metrics:')
        self.metrics.print_metrics(metrics)
        self.metrics.plot_confusion_matrix(y_test, y_pred)

if __name__ == '__main__':
    runner = EvaluateRunner()
    data_dir = 'data/processed/csl_isolated'
    model_type = input('请选择模型类型 (svm/rf/mlp/lstm/transformer): ')
    runner.run(data_dir, model_type)