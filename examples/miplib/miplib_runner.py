import torch
import time
import numpy as np
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from .data_loader import DataLoaderTrait
from .model import ModelTrait
from .config import ConfigTrait
from .naive_model import NaiveModelTrait
from .neural_network_model import NeuralNetworkModelTrait

class MIPLIBRunner(DataLoaderTrait, ModelTrait, ConfigTrait, NaiveModelTrait, NeuralNetworkModelTrait):
    def __init__(self):
        ConfigTrait.__init__(self)
        NaiveModelTrait.__init__(self)
        self.config = self.get_config()
        self.device = torch.device(self.config['device'])
        self.set_seed(41)
        self.num_classes = None
        self.num_node_features = None
        self.models = []

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to {seed}")

    def run(self):
        dataset, data = self.load_data(
            root=self.config['root'],
            instance_limit=self.config['instance_limit'],
            max_edges=self.config['max_edges'],
            max_constraints=self.config['max_constraints']
        )
        self.num_node_features = dataset.num_features
        self.num_classes = data.y.shape[0]  

        self.models = [
            self.create_naive_model(),
            self.create_neural_network_model()
        ]

        for model in self.models:
            self.run_model(model)

    def run_model(self, model):
        print(f"Running {model.name} Model...")
        model.fit_and_evaluate(self.train_loader, self.test_loader, self.config)
        
    def print_metrics(self, true, pred):
        accuracy = (np.array(true) == np.array(pred)).mean()
        mae = np.mean(np.abs(np.array(true) - np.array(pred)))
        mse = np.mean((np.array(true) - np.array(pred))**2)
        print(f'Accuracy: {accuracy:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}')

    def print_confusion_matrix(self, true, pred):
        cm = confusion_matrix(true, pred)
        print("Confusion Matrix:")
        print(cm)

    def print_sample_predictions(self, true, pred, n=10):
        print(f"Sample Predictions (first {n}):")
        for t, p in zip(true[:n], pred[:n]):
            print(f"True: {t}, Predicted: {p}")
        print()

    def print_epoch_results(self, epoch, train_loss, train_acc, test_acc, epoch_time, train_mae, train_mse, test_mae, test_mse, test_true, test_pred):
        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, '
              f'Time: {epoch_time:.2f}s')
        print(f'Train MAE: {train_mae:.4f}, Train MSE: {train_mse:.4f}')
        print(f'Test MAE: {test_mae:.4f}, Test MSE: {test_mse:.4f}')
        
        cm = confusion_matrix(test_true, test_pred)
        print("Confusion Matrix (Test Set):")
        print(cm)
        
        print("Sample Predictions (Test Set):")
        for true, pred in zip(test_true[:10], test_pred[:10]):
            print(f"True: {true}, Predicted: {pred}")
        
        print("\n")

    def print_distribution(self, train_pred, test_pred, train_true, test_true):
        train_pred_dist = np.bincount(train_pred)
        test_pred_dist = np.bincount(test_pred)
        print("Distribution of predictions (Train Set):")
        print(train_pred_dist)
        print("Distribution of predictions (Test Set):")
        print(test_pred_dist)

        train_true_dist = np.bincount(train_true)
        test_true_dist = np.bincount(test_true)
        print("Distribution of true values (Train Set):")
        print(train_true_dist)
        print("Distribution of true values (Test Set):")
        print(test_true_dist)

if __name__ == "__main__":
    runner = MIPLIBRunner()
    runner.run()