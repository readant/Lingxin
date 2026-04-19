import matplotlib.pyplot as plt
import numpy as np

class Visualization:
    def __init__(self):
        pass
    
    def plot_landmarks(self, landmarks, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        for hand_landmarks in landmarks:
            for connection in connections:
                start = hand_landmarks[connection[0]]
                end = hand_landmarks[connection[1]]
                ax.plot([start[0], end[0]], [start[1], end[1]], 'r-')
            
            ax.scatter(hand_landmarks[:, 0], hand_landmarks[:, 1], s=20, c='b')
        
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        return ax
    
    def plot_accuracy(self, history):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy')
        plt.legend()
        plt.show()
    
    def plot_loss(self, history):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Model Loss')
        plt.legend()
        plt.show()