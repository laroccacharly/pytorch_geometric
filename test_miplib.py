import os 
from torch_geometric.datasets.miplib import MIPLIB 
import torch

def test_miplib():
    # Use a directory in the current folder
    root = './data'
    os.makedirs(root, exist_ok=True)

    print("Initializing MIPLIB dataset...")
    dataset = MIPLIB(root, instance_limit=5, force_reload=True)  # Limit to 5 instances for testing

    print(f"Dataset length: {len(dataset)}")
    assert len(dataset) > 0, "Dataset should not be empty"

    print("Accessing first graph...")
    data = dataset[0]

    print(f"Number of classes: {dataset.num_classes}")
    assert dataset.num_classes > 0, "Dataset should have classes"
    assert data.y.size() == (dataset.num_classes,), "Label should be a multi-hot encoded vector"
    assert data.y.dtype == torch.float, "Label should be a float tensor"
    assert (data.y >= 0).all() and (data.y <= 1).all(), "Label values should be between 0 and 1"

    print(f"Number of nodes in first graph: {data.num_nodes}")
    print(f"Number of edges in first graph: {data.num_edges}")
    print(f"Node feature dimensions: {data.x.size(1)}")
    print(f"Number of classes: {dataset.num_classes}")

    assert isinstance(data.x, torch.Tensor), "Node features should be a tensor"
    assert isinstance(data.edge_index, torch.Tensor), "Edge index should be a tensor"
    assert isinstance(data.y, torch.Tensor), "Labels should be a tensor"

    assert data.x.dim() == 2, "Node features should be 2-dimensional"
    assert data.edge_index.dim() == 2, "Edge index should be 2-dimensional"
    assert data.edge_index.size(0) == 2, "Edge index should have 2 rows"
    assert data.y.dim() == 1, "Labels should be 1-dimensional"

    assert data.num_nodes == data.x.size(0), "Number of nodes should match feature matrix size"
    assert data.num_edges == data.edge_index.size(1), "Number of edges should match edge index size"

    print("All tests passed successfully!")


if __name__ == "__main__":
    test_miplib()

