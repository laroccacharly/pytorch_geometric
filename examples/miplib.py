import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import MIPLIB
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the MIPLIB dataset
dataset = MIPLIB(root='data', instance_limit=10, force_reload=True, max_edges=5000)
data = dataset[0]

# After loading the dataset
print(f"Shape of data.y: {data.y.shape}")

# Split the dataset into train and test
train_dataset = dataset[:int(len(dataset)*0.8)]
test_dataset = dataset[int(len(dataset)*0.8):]

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the GNN model
class Net(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.out = torch.nn.Linear(32, num_classes)  # Changed from 64 to 32
        print(f"Initialized Net with num_node_features: {num_node_features}, num_classes: {num_classes}")

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.out(x)
        return x  # No activation, as we'll use BCEWithLogitsLoss

# When initializing the model:
num_node_features = dataset.num_features
num_classes = data.y.shape[0]
print(f"Dataset info - num_features: {num_node_features}, num_classes: {num_classes}")
model = Net(num_node_features, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        # Ensure the target is a 2D tensor of float type
        target = data.y.to(torch.float)
        # There is an issue where labels are flattened into a 1D tensor
        # Need to reshape target to match the shape of out
        target = target.view(out.shape)
        
        loss = F.binary_cross_entropy_with_logits(out, target)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = (out > 0).float()  # Convert logits to binary predictions
        reshaped_y = data.y.view(pred.shape)
        correct = (pred == reshaped_y).float().sum(dim=1)
        total_correct += int(correct.sum())
        total_samples += data.num_graphs * num_classes
    return total_correct / total_samples

# Training loop
for epoch in range(1, 10):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Test Acc: {test_acc:.4f}')