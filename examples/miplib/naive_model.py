import torch.nn.functional as F
import numpy as np
from sklearn.dummy import DummyClassifier
from .targets import TargetsTrait

class NaiveModelTrait:
    def create_naive_model(self):
        return NaiveModel(self.num_classes)

class NaiveModel(NaiveModelTrait, TargetsTrait):
    def __init__(self, num_classes):
        super().__init__()
        self.name = "Naive"
        self.model = DummyClassifier(strategy='most_frequent')
        self.num_classes = num_classes

    def fit_and_evaluate(self, train_loader, test_loader, config):
        print("Fitting Naive Model...")
        self.fit(train_loader)
        
        print("Evaluating Naive Model...")
        train_acc, train_mae, train_mse, _, _ = self.evaluate(train_loader)
        test_acc, test_mae, test_mse, _, _ = self.evaluate(test_loader)
        
        print("Naive Model Results:")
        print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        print(f'Train MAE: {train_mae:.4f}, Train MSE: {train_mse:.4f}')
        print(f'Test MAE: {test_mae:.4f}, Test MSE: {test_mse:.4f}')

    def fit(self, train_loader):
        all_y = []
        for batch in train_loader:
            y = self.get_targets_from_batch_as_count(batch).numpy()
            all_y.extend(y)
        self.model.fit(np.zeros((len(all_y), 1)), all_y)

    def evaluate(self, loader):
        true, pred = self.get_predictions(loader)
        accuracy = (np.array(true) == np.array(pred)).mean()
        mae = np.mean(np.abs(np.array(true) - np.array(pred)))
        mse = np.mean((np.array(true) - np.array(pred))**2)
        return accuracy, mae, mse, true, pred

    def get_predictions(self, loader):
        all_true = []
        all_pred = []
        for batch in loader:
            y_true = self.get_targets_from_batch_as_count(batch).numpy()
            y_pred = self.model.predict(np.zeros((len(y_true), 1)))
            all_true.extend(y_true)
            all_pred.extend(y_pred)
        return all_true, all_pred
