import torch
import torch.nn as nn
import torch.optim as optim

class TransformerModel(nn.Module):
    def __init__(self, input_shape, num_classes, d_model=64, num_heads=2, dff=128, num_layers=2, dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        self.input_size = input_shape[1]
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = nn.Linear(self.input_size, d_model)
        self.pos_encoding = self._generate_positional_encoding(input_shape[0], d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dff,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        self.fc1 = nn.Linear(d_model, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def _generate_positional_encoding(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe
    
    def forward(self, x):
        x = self.embedding(x)
        x = x * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)
        
        out = self.transformer_encoder(x)
        out = out.mean(dim=1)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    
    def train_model(self, train_loader, val_loader=None, epochs=50, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
            
            if val_loader:
                self.eval()
                val_loss = 0.0
                correct = 0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        outputs = self(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs, 1)
                        correct += (predicted == labels).sum().item()
                
                val_loss = val_loss / len(val_loader.dataset)
                val_accuracy = correct / len(val_loader.dataset)
                print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            _, predicted = torch.max(outputs, 1)
        return predicted
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))