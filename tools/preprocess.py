import os
import numpy as np
from src.utils.data_loader import DataLoader
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self):
        self.loader = DataLoader()
        self.scaler = StandardScaler()
    
    def preprocess(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        X, y, class_labels = self.loader.load_data(input_dir)
        X_scaled = self.scaler.fit_transform(X)
        
        np.save(os.path.join(output_dir, 'X.npy'), X_scaled)
        np.save(os.path.join(output_dir, 'y.npy'), y)
        np.save(os.path.join(output_dir, 'class_labels.npy'), class_labels)
        
        print(f'Preprocessed data saved to {output_dir}')
        print(f'Number of samples: {len(X)}')
        print(f'Number of classes: {len(class_labels)}')
    
    def preprocess_sequence(self, input_dir, output_dir, max_length=30):
        os.makedirs(output_dir, exist_ok=True)
        
        X, y, class_labels = self.loader.load_sequence_data(input_dir, max_length)
        
        for i in range(X.shape[2]):
            X[:, :, i] = self.scaler.fit_transform(X[:, :, i])
        
        np.save(os.path.join(output_dir, 'X_sequence.npy'), X)
        np.save(os.path.join(output_dir, 'y_sequence.npy'), y)
        np.save(os.path.join(output_dir, 'class_labels.npy'), class_labels)
        
        print(f'Preprocessed sequence data saved to {output_dir}')
        print(f'Number of samples: {len(X)}')
        print(f'Number of classes: {len(class_labels)}')

if __name__ == '__main__':
    preprocessor = Preprocessor()
    input_dir = 'data/raw/collected'
    output_dir = 'data/processed/csl_isolated'
    preprocessor.preprocess(input_dir, output_dir)