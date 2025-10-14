import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from imblearn.over_sampling import SMOTE
import joblib

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
        print(f"Using device: {self.device}")
        
    def load_and_preprocess_data(self):
        """Load and preprocess data for SNN"""
        print("Loading data...")
        X = pd.read_csv(self.X_path)
        y = pd.read_csv(self.y_path)['Condition']
        
        # Filter for AD vs HC3
        mask = y.isin(['AD', 'HC3'])
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        
        print(f"Data shape after filtering: {X.shape}")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Encode labels
        y = (y == 'HC3').astype(int)
        
        # Apply SMOTE
        print("Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        print(f"Data shape after SMOTE: {X.shape}")
        
        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y.values, dtype=torch.long), scaler
    
    def train_snn(self, num_epochs=100):
        """Train the spiking neural network"""
        X, y, scaler = self.load_and_preprocess_data()
        
        # Save scaler for later use
        joblib.dump(scaler, 'scaler.pkl')
        print("Saved scaler to scaler.pkl")
        
        # Convert to PyTorch datasets
        dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        input_size = X.shape[1]
        model = SNNModel(input_size=input_size, hidden_size=64, output_size=2)
        model.to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        
        print("Starting training...")
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            batch_count = 0
            
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
                batch_count += 1
            
            if epoch % 10 == 0:
                accuracy = self.evaluate_snn(model, X, y)
                avg_loss = total_loss / batch_count if batch_count > 0 else total_loss
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        final_accuracy = self.evaluate_snn(model, X, y)
        print(f"Training completed. Final accuracy: {final_accuracy:.4f}")
        
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

def convert_for_spinnaker(model):
    """Convert PyTorch SNN to SpiNNaker-compatible format"""
    print("Converting model for SpiNNaker...")
    
    # Extract weights and biases
    weights = []
    biases = []
    
    for layer in [model.fc1, model.fc2, model.fc3]:
        weights.append(layer.weight.detach().cpu().numpy())
        biases.append(layer.bias.detach().cpu().numpy())
    
    # Convert to SpiNNaker population parameters
    population_params = {
        'n_neurons': [model.fc1.out_features, model.fc2.out_features, model.fc3.out_features],
        'weights': weights,
        'biases': biases,
        'tau_mem': [model.lif1.beta, model.lif2.beta, model.lif3.beta],  # Membrane time constants
        'input_size': model.fc1.in_features,
        'output_size': model.fc3.out_features
    }
    
    print(f"Model converted: {population_params['n_neurons']} neurons per layer")
    return population_params

def save_spinnaker_params(spinnaker_params, filename="spinnaker_model.pkl"):
    """Save SpiNNaker parameters to file"""
    joblib.dump(spinnaker_params, filename)
    print(f"Saved SpiNNaker parameters to {filename}")

# Example usage
if __name__ == "__main__":
    # Train SNN
    neuromorphic_trainer = NeuromorphicTrainer()
    trained_snn = neuromorphic_trainer.train_snn(num_epochs=50)
    
    # Convert for SpiNNaker
    spinnaker_params = convert_for_spinnaker(trained_snn)
    
    # Save parameters
    save_spinnaker_params(spinnaker_params)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("Next step: Run deploy_snn.py on SpiNNaker hardware/simulator")
    print("="*50)
