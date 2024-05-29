import os
from typing import Optional, Callable, List

import networkx as nx
import torch
import tqdm
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected


# todo: also allow a list of names so I can preprocess and load all data at once

class SR25Dataset(Dataset):
    """
        SR25: Set of strongly regular graphs on 16-40 vertices
        Source: https://users.cecs.anu.edu.au/~bdm/data/graphs.html, but not sure where the graph6 format ones are from,
        I got them from: https://github.com/gasmichel/PathNNs_expressive/tree/main/synthetic/data/SR25
    """

    def __init__(self, raw_file_name, root="data/SR25/", name="sr16622", length=5,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.root = root
        self.name = name
        self.raw_file_name = raw_file_name 
        self.length = length
        super().__init__(root, transform, pre_transform, pre_filter)

        #path = self.processed_paths[0]
        #self.data, self.slices = torch.load(path)

    def len(self):
        return len(nx.read_graph6(self.raw_file_name))

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{self.name}_path-{self.length}-{idx}.pt'))
        return data

    @property
    def raw_file_names(self) -> List[str]:
        return [f'{self.name}.txt']

    @property
    def processed_file_names(self) -> List[str]:
        data_list = []
        for idx in range(self.len()):
            data_list.append(f'data_{self.name}_path-{self.length}-{idx}.pt')
        return data_list

    def process(self):

        data_list = []

        dataset = nx.read_graph6(self.raw_file_name)

        for i, g in enumerate(tqdm.tqdm(dataset)):
            x = torch.ones(g.number_of_nodes(), 1)
            edge_index = to_undirected(torch.tensor(list(g.edges())).transpose(1, 0))
            data = Data(edge_index=edge_index, x=x, y=torch.tensor(0), num_nodes=g.number_of_nodes())
            data.graph_indicator = i

            if self.pre_filter is not None:
                data = self.pre_filter(data)
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
            torch.save(data, os.path.join(self.processed_dir, f'data_{self.name}_path-{self.length}-{i}.pt'))


        #torch.save(self.collate(data_list), self.processed_paths[0])
