import numpy as np
from sklearn.dummy import DummyClassifier

class NaiveModel():
    def __init__(self):
        self.name = "Naive"
        self.naive = DummyClassifier(strategy='most_frequent')


    def fit_and_evalute_naive(self):
        print("Fitting Naive Model...")
        self.fit_naive()
        
        print("Evaluating Naive Model...")
        train_acc, train_mae, train_mse, _, _ = self.evaluate_naive(self.train_loader)
        test_acc, test_mae, test_mse, _, _ = self.evaluate_naive(self.test_loader)       


    def fit_naive(self):
        all_y = []
        for batch in self.train_loader:
            y = self.get_targets_from_batch_as_count(batch).numpy()
            all_y.extend(y)
        self.naive.fit(np.zeros((len(all_y), 1)), all_y)

    def evaluate_naive(self, loader):
        true, pred = self.get_predictions(loader)
        accuracy = (np.array(true) == np.array(pred)).mean()
        mae = np.mean(np.abs(np.array(true) - np.array(pred)))
        mse = np.mean((np.array(true) - np.array(pred))**2)
        print("Naive Model Results:")
        print("Accuracy: ", accuracy)
        return accuracy, mae, mse, true, pred

    def get_predictions(self, loader):
        all_true = []
        all_pred = []
        for batch in loader:
            y_true = self.get_targets_from_batch_as_count(batch).numpy()
            y_pred = self.naive.predict(np.zeros((len(y_true), 1)))
            all_true.extend(y_true)
            all_pred.extend(y_pred)
        return all_true, all_pred
