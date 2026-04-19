from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Metrics:
    def __init__(self):
        pass
    
    def calculate_metrics(self, y_true, y_pred):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        if class_names:
            plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
            plt.yticks(np.arange(len(class_names)), class_names, rotation=0)
        plt.title('Confusion Matrix')
        plt.show()
    
    def print_metrics(self, metrics):
        for key, value in metrics.items():
            print(f'{key}: {value:.4f}')