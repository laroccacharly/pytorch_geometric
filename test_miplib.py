import os
import os.path as osp
import zipfile
from typing import List, Optional
import pandas as pd

import torch
from torch_geometric.data import Data, InMemoryDataset, download_url

import pulp
import gzip


class MIPLIB(InMemoryDataset):
    url = 'https://miplib.zib.de/downloads/benchmark.zip'
    url_csv = "https://raw.githubusercontent.com/laroccacharly/pytorch_geometric/miplib_benchmark_data/miplib_benchmark.csv"
    csv_headers=["InstanceInst.","StatusStat.","VariablesVari.","BinariesBina.","IntegersInte.","ContinuousCont.","ConstraintsCons.","Nonz.Nonz.","SubmitterSubm.","GroupGrou.","ObjectiveObje.","TagsTags."]
    
    def __init__(self, root: str, instance_limit: Optional[int] = None, force_reload: bool = False):
        self.instance_limit = instance_limit
        self.tag_to_label = {}
        self._num_classes = 0  # Use a private attribute
        super().__init__(root, transform=None, pre_transform=None, force_reload=force_reload)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value

    @property
    def raw_file_names(self) -> List[str]:
        return ['benchmark.zip']

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.instance_limit}.pt'

    def download(self):
        self._download_zip()
        self._download_csv()

    def _download_zip(self):
        download_url(self.url, self.raw_dir)
        print(f"Downloaded zip file to {self.raw_dir}")

    def _download_csv(self):
        csv_path = osp.join(self.raw_dir, 'miplib_benchmark.csv')
        download_url(self.url_csv, self.raw_dir)
        print(f"Downloaded CSV file to {csv_path}")

    def process(self):
        self._extract_zip()
        self._extract_gz_files()
        data_list = self._process_mps_files()
        self._save_processed_data(data_list)

    def _extract_zip(self):
        zip_path = osp.join(self.raw_dir, 'benchmark.zip')
        extract_dir = self.raw_dir
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted zip file to {extract_dir}")

    def _extract_gz_files(self):
        csv_data = pd.read_csv(osp.join(self.raw_dir, 'miplib_benchmark.csv'))
        instances_to_process = csv_data['InstanceInst.'].head(self.instance_limit) if self.instance_limit else csv_data['InstanceInst.']

        for filename in os.listdir(self.raw_dir):
            if filename.endswith('.mps.gz'):
                instance_name = filename[:-7]  # Remove .mps.gz extension
                if instance_name in instances_to_process.values:
                    gz_path = osp.join(self.raw_dir, filename)
                    mps_path = gz_path[:-3]  # Remove .gz extension
                    if not osp.exists(mps_path):
                        with gzip.open(gz_path, 'rb') as f_in:
                            with open(mps_path, 'wb') as f_out:
                                f_out.write(f_in.read())
                        print(f"Extracted: {filename}")
                    else:
                        print(f"Skipped extraction (already exists): {filename}")

    def _process_mps_files(self):
        data_list = []
        mps_dir = self.raw_dir
        csv_data = pd.read_csv(osp.join(self.raw_dir, 'miplib_benchmark.csv'))
        instances_to_process = csv_data.head(self.instance_limit) if self.instance_limit else csv_data

        # Create tag-to-label mapping
        all_tags = set()
        for tags in instances_to_process['TagsTags.']:
            all_tags.update(tags.split())  # Split by whitespace instead of comma
        self.tag_to_label = {tag: i for i, tag in enumerate(sorted(all_tags))}
        self.num_classes = len(self.tag_to_label)

        # Print all tags found
        print("All tags found:")
        for tag in sorted(all_tags):
            print(f"  - {tag}")

        for _, row in instances_to_process.iterrows():
            instance_name = row['InstanceInst.']
            filename = f"{instance_name}.mps"
            filepath = osp.join(mps_dir, filename)
            
            if osp.exists(filepath):
                print(f"Processing file: {filename}")
                tags = row['TagsTags.']
                data = self._process_mps_file(filepath, tags)
                data_list.append(data)
            else:
                print(f"Warning: MPS file not found for instance {instance_name}")

        print(f"Processed {len(data_list)} instances")
        return data_list

    def _save_processed_data(self, data_list):
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _process_mps_file(self, mps_file_path, tags):
        instance = pulp.LpProblem.fromMPS(mps_file_path)
        x = self._extract_features(instance)
        edge_index = self._create_edge_index(instance)
        y = self._get_label(tags)
        return Data(x=x, edge_index=edge_index, y=y)

    def _extract_features(self, problem):
        # Placeholder: Extract relevant features from the PuLP problem
        return torch.rand(10, 5)  # Random features for debugging

    def _create_edge_index(self, problem):
        # Placeholder: Create edge index based on the problem structure
        return torch.randint(0, 10, (2, 15))  # Random edges for debugging

    def _get_label(self, tags):
        label = torch.zeros(self.num_classes, dtype=torch.float)
        print(f"Tags for this instance: {tags}")
        for tag in tags.split():  # Split by whitespace instead of comma
            if tag in self.tag_to_label:
                label[self.tag_to_label[tag]] = 1.0
            else:
                print(f"Warning: Tag '{tag}' not found in tag_to_label mapping")
        return label

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)}, num_classes={self.num_classes})'


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

