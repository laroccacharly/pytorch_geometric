import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import MIPLIB
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix
import numpy as np

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the MIPLIB dataset
dataset = MIPLIB(root='data', instance_limit=60, max_edges=2000)
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
        return x  

# When initializing the model:
num_node_features = dataset.num_features
num_classes = data.y.shape[0]
print(f"Dataset info - num_features: {num_node_features}, num_classes: {num_classes}")
model = Net(num_node_features, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    total_loss = 0
    all_true = []
    all_pred = []
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        
        true_tag_count = batch.y.view(out.shape).sum(dim=1).long()
        loss = F.cross_entropy(out, true_tag_count)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        
        all_true.extend(true_tag_count.cpu().numpy())
        all_pred.extend(out.argmax(dim=1).cpu().numpy())
    
    mae = mean_absolute_error(all_true, all_pred)
    mse = mean_squared_error(all_true, all_pred)
    return total_loss / len(train_loader.dataset), mae, mse, all_true, all_pred

@torch.no_grad()
def test(loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_true = []
    all_pred = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        
        true_tag_count = batch.y.view(out.shape).sum(dim=1).long()
        predicted_tag_count = out.argmax(dim=1)
        
        total_correct += (predicted_tag_count == true_tag_count).sum().item()
        total_samples += true_tag_count.size(0)
        
        all_true.extend(true_tag_count.cpu().numpy())
        all_pred.extend(predicted_tag_count.cpu().numpy())
    
    accuracy = total_correct / total_samples
    mae = mean_absolute_error(all_true, all_pred)
    mse = mean_squared_error(all_true, all_pred)
    return accuracy, mae, mse, all_true, all_pred

# Training loop
max_epochs = 1000
for epoch in range(1, max_epochs + 1):
    train_loss, train_mae, train_mse, train_true, train_pred = train()
    train_acc, _, _, _, _ = test(train_loader)
    test_acc, test_mae, test_mse, test_true, test_pred = test(test_loader)
    
    if epoch % 50 == 0 or epoch == max_epochs:
        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        print(f'Train MAE: {train_mae:.4f}, Train MSE: {train_mse:.4f}')
        print(f'Test MAE: {test_mae:.4f}, Test MSE: {test_mse:.4f}')
        
        # Print confusion matrix for test set
        cm = confusion_matrix(test_true, test_pred)
        print("Confusion Matrix (Test Set):")
        print(cm)
        
        # Print some sample predictions
        print("Sample Predictions (Test Set):")
        for true, pred in zip(test_true[:10], test_pred[:10]):
            print(f"True: {true}, Predicted: {pred}")
        
        print("\n")

# After training, print the distribution of predictions
train_pred_dist = np.bincount(train_pred)
test_pred_dist = np.bincount(test_pred)
print("Distribution of predictions (Train Set):")
print(train_pred_dist)
print("Distribution of predictions (Test Set):")
print(test_pred_dist)

# Print the distribution of true values
train_true_dist = np.bincount(train_true)
test_true_dist = np.bincount(test_true)
print("Distribution of true values (Train Set):")
print(train_true_dist)
print("Distribution of true values (Test Set):")
print(test_true_dist)