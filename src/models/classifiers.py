from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

class Classifiers:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'svm': SVC(kernel='rbf', C=1.0, gamma='scale'),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
        }
    
    def train(self, X_train, y_train, model_name='svm'):
        X_train_scaled = self.scaler.fit_transform(X_train)
        model = self.models.get(model_name)
        if model:
            model.fit(X_train_scaled, y_train)
            return model
        else:
            raise ValueError(f"Model {model_name} not found")
    
    def predict(self, X, model, model_name='svm'):
        X_scaled = self.scaler.transform(X)
        return model.predict(X_scaled)
    
    def get_model(self, model_name):
        return self.models.get(model_name)