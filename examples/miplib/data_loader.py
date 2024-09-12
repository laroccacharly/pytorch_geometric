from torch_geometric.datasets import MIPLIB
from torch_geometric.loader import DataLoader

class DataLoaderTrait:
    def load_data(self, root, instance_limit, max_edges, max_constraints):
        print("Loading data...")
        dataset = MIPLIB(root=root, instance_limit=instance_limit, max_edges=max_edges, max_constraints=max_constraints)
        train_dataset = dataset[:int(len(dataset)*0.8)]
        test_dataset = dataset[int(len(dataset)*0.8):]
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        return dataset, dataset[0]