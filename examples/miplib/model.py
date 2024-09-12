import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class ModelTrait:
    class Net(torch.nn.Module):
        def __init__(self, num_node_features, num_classes, hidden_channels):
            super().__init__()
            self.conv1 = GCNConv(num_node_features, hidden_channels[0])
            self.conv2 = GCNConv(hidden_channels[0], hidden_channels[1])
            self.out = torch.nn.Linear(hidden_channels[1], num_classes)

        def forward(self, x, edge_index, batch):
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = global_mean_pool(x, batch)
            x = self.out(x)
            return x

    def init_model(self, num_node_features, num_classes, device):
        print("Initializing model...")
        self.model = self.Net(num_node_features, num_classes, self.config['hidden_channels']).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])