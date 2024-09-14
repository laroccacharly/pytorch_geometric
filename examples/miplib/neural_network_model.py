import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import mean_absolute_error, mean_squared_error

class NNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "Neural Network"
        hidden_channels = self.config['hidden_channels']
        num_node_features = self.num_node_features

        self.conv1 = GCNConv(num_node_features, hidden_channels[0])
        self.conv2 = GCNConv(hidden_channels[0], hidden_channels[1])
        self.out = torch.nn.Linear(hidden_channels[1], self.num_classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.out(x)
        return x
    
    def fit_and_evaluate_nn(self):
        print("Fitting Neural Network Model...")
        self.fit_nn()
        
        print("Evaluating Neural Network Model...")
        train_acc, train_mae, train_mse, _, _ = self.evaluate_nn(self.train_loader)
        test_acc, test_mae, test_mse, _, _ = self.evaluate_nn(self.test_loader) 
        
    def train_step(self, batch):
        self.train()
        self.optimizer.zero_grad()
        out = self.forward(batch.x, batch.edge_index, batch.batch)
        targets = self.get_targets_from_batch_as_one_hot(batch)
        loss = F.cross_entropy(out, targets)
        loss.backward()
        self.optimizer.step()
        return loss

    def fit_nn(self):
        max_epochs = 100
        loader = self.train_loader
        best_test_acc = 0
        patience = self.config['patience']
        counter = 0
        early_stop = False
        for epoch in range(1, max_epochs + 1):
            loss = self.fit_one_epoch(loader)
            train_accuracy, _, _, _, _ = self.evaluate_nn(loader)
            test_accuracy, _, _, _, _ = self.evaluate_nn(self.test_loader)
            print(f'Epoch {epoch}, Loss: {loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                counter = 0
            else:
                counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                early_stop = True

            if epoch % 50 == 0 or epoch == max_epochs or early_stop:
                test_acc, _, _, _, _ = self.evaluate_nn(self.test_loader)
                print(f'Epoch {epoch}, Test Accuracy: {test_acc:.4f}')

            if early_stop: 
                break 
    
    def fit_one_epoch(self, loader):
        total_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            loss = self.train_step(batch)
            total_loss += loss.item() * batch.num_graphs

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate_nn(self, loader):
        self.eval()
        total_correct = 0
        total_samples = 0
        all_true = []
        all_pred = []
        for batch in loader:
            batch = batch.to(self.device)
            out = self(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            target = self.get_targets_from_batch_as_count(batch)
            total_correct += (pred == target).sum()
            total_samples += batch.num_graphs
            all_true.extend(target.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
        
        accuracy = total_correct / total_samples
        mae = mean_absolute_error(all_true, all_pred)
        mse = mean_squared_error(all_true, all_pred)
        print("NN results" )
        print("accuracy: ", accuracy)
        return accuracy, mae, mse, all_true, all_pred