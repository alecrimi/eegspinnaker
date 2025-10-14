import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class SNNModel(nn.Module):
    """Spiking Neural Network compatible with neuromorphic hardware"""
    
    def __init__(self, input_size, hidden_size, output_size, beta=0.5):
        super(SNNModel, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
        self.num_steps = 25  # Time steps
        self.hidden_size = hidden_size
        
    def forward(self, x):
        # Initialize hidden states and outputs
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        spk3_rec = []  # Record output spikes
        mem3_rec = []  # Record output membrane potentials
        
        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)
            
        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)

class NeuromorphicTrainer:
    def __init__(self, X_path="data/X_ml.csv", y_path="data/y_ml.csv"):
        self.X_path = X_path
        self.y_path = y_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_and_preprocess_data(self):
        """Load and preprocess data for SNN"""
        X = pd.read_csv(self.X_path)
        y = pd.read_csv(self.y_path)['Condition']
        
        # Filter for AD vs HC3
        mask = y.isin(['AD', 'HC3'])
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        
        # Encode labels
        y = (y == 'HC3').astype(int)
        
        # Apply SMOTE
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        
        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y.values, dtype=torch.long)
    
    def train_snn(self, num_epochs=100):
        """Train the spiking neural network"""
        X, y = self.load_and_preprocess_data()
        
        # Convert to PyTorch datasets
        dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        input_size = X.shape[1]
        model = SNNModel(input_size=input_size, hidden_size=64, output_size=2)
        model.to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                spk_rec, mem_rec = model(batch_X)
                
                # Sum spikes over time
                output = torch.sum(spk_rec, dim=0)
                
                loss = loss_fn(output, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                accuracy = self.evaluate_snn(model, X, y)
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return model
    
    def evaluate_snn(self, model, X, y):
        """Evaluate SNN performance"""
        model.eval()
        with torch.no_grad():
            X, y = X.to(self.device), y.to(self.device)
            spk_rec, mem_rec = model(X)
            output = torch.sum(spk_rec, dim=0)
            predictions = torch.argmax(output, dim=1)
            accuracy = (predictions == y).float().mean()
        return accuracy.item()
