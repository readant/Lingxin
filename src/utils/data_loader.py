import os
import numpy as np

class DataLoader:
    def __init__(self):
        pass
    
    def load_data(self, data_dir):
        X = []
        y = []
        class_labels = {}
        label_idx = 0
        
        for word_dir in os.listdir(data_dir):
            word_path = os.path.join(data_dir, word_dir)
            if not os.path.isdir(word_path):
                continue
            
            if word_dir not in class_labels:
                class_labels[word_dir] = label_idx
                label_idx += 1
            
            for file_name in os.listdir(word_path):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(word_path, file_name)
                    data = np.load(file_path)
                    X.append(data)
                    y.append(class_labels[word_dir])
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y, class_labels
    
    def load_sequence_data(self, data_dir, max_length=30):
        X = []
        y = []
        class_labels = {}
        label_idx = 0
        
        for word_dir in os.listdir(data_dir):
            word_path = os.path.join(data_dir, word_dir)
            if not os.path.isdir(word_path):
                continue
            
            if word_dir not in class_labels:
                class_labels[word_dir] = label_idx
                label_idx += 1
            
            for file_name in os.listdir(word_path):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(word_path, file_name)
                    data = np.load(file_path)
                    
                    if len(data) < max_length:
                        pad = np.zeros((max_length - len(data), data.shape[1]))
                        data = np.vstack([data, pad])
                    else:
                        data = data[:max_length]
                    
                    X.append(data)
                    y.append(class_labels[word_dir])
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y, class_labels