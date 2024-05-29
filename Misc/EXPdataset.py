# These graphs are all 1-WL equivalent

# raw data from here: https://github.com/ralphabb/GNN-RNI/tree/main/Data
import os
from collections import defaultdict
from typing import Optional, Callable, List

import networkx as nx
import numpy as np
import torch
import tqdm
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx


class EXPDataset(InMemoryDataset):
    def __init__(self, raw_file_name, root="data/EXP", name='EXP',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 task="iso", length=5, n_splits=4):
        self.name = name
        self.length = length
        self.raw_file_name = raw_file_name
        self.task = task
        self.n_splits = n_splits
        super().__init__(root, transform, pre_transform, pre_filter)

        path = self.processed_paths[0]

        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return [f'{self.name}.txt']

    @property
    def processed_file_names(self) -> List[str]:
        return [f'data_path-{self.length}.pt']

    def process(self):
        data_list = []
        with open(self.raw_file_name, "r") as data:
            num_graphs = int(data.readline().rstrip().split(" ")[0])
            for _ in tqdm.tqdm(range(num_graphs)):
                graph_meta = data.readline().rstrip().split(" ")
                num_vertex = int(graph_meta[0])
                curr_graph = np.zeros(shape=(num_vertex, num_vertex))
                for j in range(num_vertex):
                    vertex = data.readline().rstrip().split(" ")
                    for k in range(2, len(vertex)):
                        curr_graph[j, int(vertex[k])] = np.ones((1))
                g = from_networkx(nx.from_numpy_array(curr_graph))
                g.y = torch.unsqueeze(torch.tensor(float(graph_meta[1])), dim=0)
                g.x = torch.ones((num_vertex, 1))

                if self.pre_filter is not None:
                    g = self.pre_filter(g)
                if self.pre_transform is not None:
                    g = self.pre_transform(g)

                data_list.append(g)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def get_all_splits_idx(self):
        splits = defaultdict(list)
        val_size = int(self.__len__() * 0.1)
        test_size = int(self.__len__() * 0.15)
        for it in range(self.n_splits):
            indices = np.arange(self.__len__())
            val_idx = np.arange(start=(it) * val_size, stop=(it + 1) * val_size)
            test_idx = np.arange(start=(it) * test_size, stop=(it + 1) * test_size)
            splits["val"].append(indices[val_idx])
            remaining_indices = np.delete(indices, val_idx)
            splits["test"].append(remaining_indices[test_idx])
            remaining_indices = np.delete(remaining_indices, test_idx)
            splits["train"].append(remaining_indices)
        return splits