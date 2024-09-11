import os
import os.path as osp
import zipfile
from typing import List, Optional
import pandas as pd

import torch
from torch_geometric.data import Data, InMemoryDataset, download_url

import pulp
import gzip
from tqdm import tqdm

class MIPLIB(InMemoryDataset):
    url = 'https://miplib.zib.de/downloads/benchmark.zip'
    url_csv = "https://raw.githubusercontent.com/laroccacharly/pytorch_geometric/miplib_benchmark_data/miplib_benchmark.csv"
    csv_headers = ["InstanceInst.","StatusStat.","VariablesVari.","BinariesBina.","IntegersInte.","ContinuousCont.","ConstraintsCons.","Nonz.Nonz.","SubmitterSubm.","GroupGrou.","ObjectiveObje.","TagsTags."]
    
    def __init__(self, root: str, instance_limit: Optional[int] = None, max_edges: int = 1000, force_reload: bool = False):
        self.instance_limit = instance_limit
        self.max_edges = max_edges
        self.tag_to_label = {}
        self._num_classes = 0  # Initialize to 0
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
        # skip extract if the files in self.raw_dir. Count the number of files in the directory and compare with the number of files in the zip file.
        # if the number of files is the same, skip the extraction.
        zip_path = osp.join(self.raw_dir, 'benchmark.zip')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            num_files_in_zip = len(zip_ref.namelist())
        
        num_files_in_dir = len(os.listdir(self.raw_dir))
        
        if num_files_in_dir >= num_files_in_zip:
            print("Skipping extraction (already extracted)")
            return
        else:
            print(f"Number of files in zip: {num_files_in_zip}")
            print(f"Number of files in dir: {num_files_in_dir}")
        extract_dir = self.raw_dir
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted zip file to {extract_dir}")

    def _get_sorted_csv_data(self):
        csv_data = pd.read_csv(osp.join(self.raw_dir, 'miplib_benchmark.csv'))
        csv_data['VariablesVari.'] = pd.to_numeric(csv_data['VariablesVari.'], errors='coerce')
        return csv_data.sort_values('VariablesVari.')

    def _extract_gz_files(self):
        csv_data = self._get_sorted_csv_data()
        instances_to_process = csv_data.head(self.instance_limit) if self.instance_limit else csv_data

        print("Extracting instances in order of increasing binary variables:")
        for _, row in instances_to_process.iterrows():
            instance_name = row['InstanceInst.']
            filename = f"{instance_name}.mps.gz"
            gz_path = osp.join(self.raw_dir, filename)
            
            if os.path.exists(gz_path):
                mps_path = gz_path[:-3]  # Remove .gz extension
                if not osp.exists(mps_path):
                    with gzip.open(gz_path, 'rb') as f_in:
                        with open(mps_path, 'wb') as f_out:
                            f_out.write(f_in.read())
                    print(f"Extracted: {filename} (Binary variables: {row['BinariesBina.']})")
                else:
                    print(f"Skipped extraction (already exists): {filename} (Binary variables: {row['BinariesBina.']})")
            else:
                print(f"Warning: GZ file not found for instance {instance_name}")

    def _process_mps_files(self):
        data_list = []
        mps_dir = self.raw_dir
        csv_data = self._get_sorted_csv_data()
        instances_to_process = csv_data.head(self.instance_limit) if self.instance_limit else csv_data

        # Create tag-to-label mapping
        all_tags = set()
        for tags in instances_to_process['TagsTags.']:
            all_tags.update(tags.split())
        self.tag_to_label = {tag: i for i, tag in enumerate(sorted(all_tags))}
        self.num_classes = len(self.tag_to_label)

        # Print all tags found
        print("All tags found:")
        for tag in sorted(all_tags):
            print(f"  - {tag}")

        print("Processing instances in order of increasing binary variables:")
        for _, row in instances_to_process.iterrows():
            print(f"  - {row['InstanceInst.']} (Binary variables: {row['BinariesBina.']})")
            instance_name = row['InstanceInst.']
            filename = f"{instance_name}.mps"
            filepath = osp.join(mps_dir, filename)
            
            if osp.exists(filepath):
                print(f"Processing file: {filename}")
                tags = row['TagsTags.']
                data = self._process_mps_file(filepath, tags)
                if data is not None:
                    data_list.append(data)
            else:
                print(f"Warning: MPS file not found for instance {instance_name}")

        print(f"Processed {len(data_list)} instances")
        return data_list

    def _process_mps_file(self, mps_file_path, tags):
        try:
            var1, instance = pulp.LpProblem.fromMPS(mps_file_path)
        except IndexError:
            print(f"Warning: Error reading MPS file {mps_file_path}. Skipping this instance.")
            return None

        x, edge_index, edge_attr = self._create_tripartite_graph(instance)
        y = self._get_label(tags)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def _create_tripartite_graph(self, instance):
        variables = instance.variables()
        constraints = instance.constraints.values()
        
        # Create node features
        var_features = self._extract_variable_features(variables)
        const_features = self._extract_constraint_features(constraints)
        obj_features = self._extract_objective_features(instance)
        
        print(f"var_features shape: {var_features.shape}")
        print(f"const_features shape: {const_features.shape}")

        # Ensure all feature tensors have the same number of features
        max_features = max(var_features.size(1), const_features.size(1), obj_features.size(1))
        
        var_features = self._pad_features(var_features, max_features)
        const_features = self._pad_features(const_features, max_features)
        obj_features = self._pad_features(obj_features, max_features)

        x = torch.cat([var_features, const_features, obj_features], dim=0)
        
        # Create edge index and edge attributes
        edge_index, edge_attr = self._create_edges(instance, len(variables), len(constraints))
        
        return x, edge_index, edge_attr

    def _pad_features(self, features, target_size):
        if features.size(1) < target_size:
            padding = torch.zeros(features.size(0), target_size - features.size(1), device=features.device)
            return torch.cat([features, padding], dim=1)
        return features

    def _extract_variable_features(self, variables):
        features = []
        for var in variables:
            feature = [
                float(var.lowBound if var.lowBound is not None else -1e9),
                float(var.upBound if var.upBound is not None else 1e9),
                float(var.cat == pulp.const.LpInteger),
                float(var.cat == pulp.const.LpBinary),
                float(var.cat == pulp.const.LpContinuous)
            ]
            features.append(feature)
        return torch.tensor(features, dtype=torch.float)

    def _extract_constraint_features(self, constraints):
        features = []
        for const in constraints:
            feature = [
                float(const.sense == pulp.const.LpConstraintLE),
                float(const.sense == pulp.const.LpConstraintGE),
                float(const.sense == pulp.const.LpConstraintEQ),
                float(const.constant)
            ]
            features.append(feature)
        return torch.tensor(features, dtype=torch.float)

    def _extract_objective_features(self, problem):
        return torch.tensor([[float(problem.sense == pulp.const.LpMaximize)]], dtype=torch.float)

    def _create_edges(self, problem, num_vars, num_constraints):
        edge_index = []
        edge_attr = []
        
        edge_count = 0
        
        with tqdm(total=self.max_edges, desc="Building edges") as pbar:
            # v-c edges
            for i, const in enumerate(problem.constraints.values()):
                for j, (var, coeff) in enumerate(const.items()):
                    if edge_count >= self.max_edges:
                        break
                    edge_index.append([j, num_vars + i])
                    edge_attr.append([float(coeff)])
                    edge_count += 1
                    pbar.update(1)
                if edge_count >= self.max_edges:
                    break
            
            # v-o edges
            if edge_count < self.max_edges:
                for i, (var, coeff) in enumerate(problem.objective.items()):
                    if edge_count >= self.max_edges:
                        break
                    edge_index.append([i, num_vars + num_constraints])
                    edge_attr.append([float(coeff)])
                    edge_count += 1
                    pbar.update(1)
            
            # c-o edges
            if edge_count < self.max_edges:
                for i in range(num_constraints):
                    if edge_count >= self.max_edges:
                        break
                    edge_index.append([num_vars + i, num_vars + num_constraints])
                    edge_attr.append([float(problem.constraints[list(problem.constraints.keys())[i]].constant)])
                    edge_count += 1
                    pbar.update(1)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        print(f"Total edges added: {edge_count}")
        return edge_index, edge_attr

    def _get_label(self, tags):
        label = torch.zeros(self.num_classes, dtype=torch.float)
        print(f"Tags for this instance: {tags}")
        for tag in tags.split():  # Split by whitespace instead of comma
            if tag in self.tag_to_label:
                label[self.tag_to_label[tag]] = 1.0
            else:
                print(f"Warning: Tag '{tag}' not found in tag_to_label mapping")
        print(f"Label shape: {label.shape}")  # Add this line
        return label

    def _save_processed_data(self, data_list):
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)}, num_classes={self.num_classes})'
