# validate_snn.py
import torch
import snntorch as snn
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def validate_on_simulation(spinnaker_params, X_test, y_test):
    """Validate that the converted parameters work in simulation"""
    
    # Recreate the SNN architecture with loaded parameters
    class ValidationSNN(torch.nn.Module):
        def __init__(self, params):
            super().__init__()
            self.fc1 = torch.nn.Linear(params['input_size'], params['n_neurons'][0])
            self.fc2 = torch.nn.Linear(params['n_neurons'][0], params['n_neurons'][1])
            self.fc3 = torch.nn.Linear(params['n_neurons'][1], params['n_neurons'][2])
            
            # Load the trained weights
            self.fc1.weight.data = torch.tensor(params['weights'][0])
            self.fc1.bias.data = torch.tensor(params['biases'][0])
            self.fc2.weight.data = torch.tensor(params['weights'][1])
            self.fc2.bias.data = torch.tensor(params['biases'][1])
            self.fc3.weight.data = torch.tensor(params['weights'][2])
            self.fc3.bias.data = torch.tensor(params['biases'][2])
            
            # Create LIF neurons with same parameters
            self.lif1 = snn.Leaky(beta=params['tau_mem'][0])
            self.lif2 = snn.Leaky(beta=params['tau_mem'][1])
            self.lif3 = snn.Leaky(beta=params['tau_mem'][2])
            self.num_steps = 25
            
        def forward(self, x):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()
            
            for step in range(self.num_steps):
                cur1 = self.fc1(x)
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                cur3 = self.fc3(spk2)
                spk3, mem3 = self.lif3(cur3, mem3)
            
            return torch.sum(spk3, dim=0)
    
    # Test the model
    model = ValidationSNN(spinnaker_params)
    model.eval()
    
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = accuracy_score(y_test.numpy(), predictions.numpy())
    
    print(f"Simulation Validation Accuracy: {accuracy:.4f}")
    return accuracy

# Add this to your train_snn.py after training
if __name__ == "__main__":
    neuromorphic_trainer = NeuromorphicTrainer()
    trained_snn = neuromorphic_trainer.train_snn(num_epochs=50)
    
    # Convert for SpiNNaker
    spinnaker_params = convert_for_spinnaker(trained_snn)
    
    # VALIDATE before saving
    X, y, scaler = neuromorphic_trainer.load_and_preprocess_data()
    val_accuracy = validate_on_simulation(spinnaker_params, X, y)
    
    if val_accuracy > 0.7:  # Your threshold
        save_spinnaker_params(spinnaker_params)
        print(f"✓ Model validated! Accuracy: {val_accuracy:.4f}")
    else:
        print(f"✗ Model validation failed! Accuracy: {val_accuracy:.4f}")
