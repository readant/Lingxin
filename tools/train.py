import numpy as np
import os
from src.training.trainer import Trainer

class TrainRunner:
    def __init__(self):
        self.trainer = Trainer()
    
    def run(self, data_dir, model_type='svm', save_dir='models'):
        os.makedirs(save_dir, exist_ok=True)
        
        if model_type in ['svm', 'rf', 'mlp']:
            X = np.load(os.path.join(data_dir, 'X.npy'))
            y = np.load(os.path.join(data_dir, 'y.npy'))
            model, accuracy = self.trainer.train_classifier(X, y, model_type)
            print(f'{model_type} model accuracy: {accuracy:.4f}')
        elif model_type == 'lstm':
            X = np.load(os.path.join(data_dir, 'X_sequence.npy'))
            y = np.load(os.path.join(data_dir, 'y_sequence.npy'))
            model, accuracy = self.trainer.train_lstm(X, y)
            model.save(os.path.join(save_dir, 'lstm_model.h5'))
            print(f'LSTM model accuracy: {accuracy:.4f}')
        elif model_type == 'transformer':
            X = np.load(os.path.join(data_dir, 'X_sequence.npy'))
            y = np.load(os.path.join(data_dir, 'y_sequence.npy'))
            model, accuracy = self.trainer.train_transformer(X, y)
            model.save(os.path.join(save_dir, 'transformer_model.h5'))
            print(f'Transformer model accuracy: {accuracy:.4f}')
        else:
            print(f'Unknown model type: {model_type}')

if __name__ == '__main__':
    runner = TrainRunner()
    data_dir = 'data/processed/csl_isolated'
    model_type = input('请选择模型类型 (svm/rf/mlp/lstm/transformer): ')
    runner.run(data_dir, model_type)