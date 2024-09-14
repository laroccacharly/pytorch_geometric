
import torch 

class ConfigTrait:
    def __init__(self):
        self.config = {
            'seed': 42,
            # Data loading parameters
            'root': 'data',
            'instance_limit': 200,
            'max_edges': 2000,
            'max_constraints': 1000,
            
            # Model parameters
            'hidden_channels': [16, 32],
            'learning_rate': 0.01,
            
            # Training parameters
            'batch_size': 32,
            'max_epochs': 100,
            'patience': 5,
            
            # Other parameters
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

    def get_config(self):
        return self.config