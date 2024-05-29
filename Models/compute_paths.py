# transform operation which precomputes the paths
# todo: two ideas: precompute using igraph
# todo: make this a proper pytorch dataset instead of extra features in normal graph

import igraph
import networkx as nx
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx

from Misc.utils import PathData


@functional_transform('compute_paths')
class ComputePaths(BaseTransform):
    def __init__(self, length, reverse=False, fast=True, max_degree=10):
        self.length = length
        self.fast = fast
        self.reverse = reverse
        self.max_deg = max_degree

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.length}, {self.reverse}, {self.fast})'

    def create_dummy_graph(self) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(list(range(4)))
        g.add_edges_from([[0, 1], [0, 2], [1, 2], [2, 3]])
        return g

    def __call__(self, data: Data) -> Data:

        g = to_networkx(data).to_undirected()

        if self.fast:
            compute_paths = self.compute_paths_faster
        else:
            compute_paths = self.compute_paths

        paths, path_mask, lengths, edges, neighbors, distances = [], [], [], [], [], []

        for node in g.nodes():
            paths_node, path_mask_node, lengths_node, edges_node, neighbors_node, distances_node = compute_paths(node, g)
            (paths.append(paths_node), path_mask.append(path_mask_node), lengths.append(lengths_node),
            edges.append(edges_node), neighbors.append(neighbors_node)), distances.append(distances_node)


        path_mask = [x for y in path_mask for x in y]
        paths = [x for y in paths for x in y]
        neighbors = [x for y in neighbors for x in y]
        distances = [x for y in distances for x in y]
        
        # Pad path length (if too short)
        if max(map(lambda d: d.shape[0], distances)) < self.length+1:
            distances[0] = torch.concat([distances[0], torch.Tensor(self.length)])
            
        lengths = [x for y in lengths for x in y]

        edges = [x for y in edges for x in y]
        # replace edges with edge attributes
        edges = [torch.stack([((data.edge_index[0] == edge[0]) & (data.edge_index[1] == edge[1])).nonzero().view(-1)[0]
                  if edge[1] > 0 else torch.tensor(-20) for edge in path]) for path in edges]

        sorted_lists = sorted(zip(paths, lengths, path_mask, edges), key=lambda x: len(x[0]), reverse=True)
        paths, lengths, path_mask, edges = map(list, zip(*sorted_lists))

        sorted_lists = sorted(zip(paths, lengths, neighbors, path_mask, edges), key=lambda x: len(x[0]), reverse=True)
        paths, lengths, neighbors, path_mask, edges = map(list, zip(*sorted_lists))
        
        if self.reverse:
            paths = [torch.flip(path, dims=(0,)) for path in paths]
            neighbors = [torch.flip(neighbor, dims=(0,)) for neighbor in neighbors]
            distances = [torch.flip(distance, dims=(0,)) for distance in distances]
            edges = [torch.flip(edge, dims=(0,)) for edge in edges]

        # hack to pad first sequence to max length to make sure all graphs are padded to max length
        paths[0] = nn.ConstantPad1d((0, self.length + 1 - paths[0].shape[0]), -10)(paths[0])
        neighbors[0] = nn.ConstantPad1d((0, self.length + 1 - neighbors[0].shape[0]), -10)(neighbors[0])
        
        padded_paths = pad_sequence(paths, batch_first=False, padding_value=-10)
        padded_neighbors = pad_sequence(neighbors, batch_first=False, padding_value=-10)
        padded_distances = pad_sequence(distances, batch_first=False, padding_value=self.length)
        mask_tensor = torch.tensor(path_mask)

        data.path_index = padded_paths
        data.path_lengths = torch.tensor(lengths)
        data.mask_index = mask_tensor

        neighbors[0] = nn.ConstantPad1d((0, self.max_deg - neighbors[0].shape[0]), -10)(neighbors[0])

        # edge stuff
        edges[0] = nn.ConstantPad1d((0, self.length + 1 - edges[0].shape[0]), -10)(edges[0])
        padded_edges = pad_sequence(edges, batch_first=False, padding_value=-10)

        # todo: shift categorical values; but this should be done somewhere else
        # data.path_edge_idx = padded_edges
        assert(len(padded_edges.shape) == 2)
        assert(padded_edges.shape[0] == self.length+1)

        
        return PathData(y = data.y, x = data.x, edge_index = data.edge_index, edge_attr = data.edge_attr, path_index = padded_paths, 
                        path_lengths = torch.tensor(lengths), mask_index = mask_tensor, path_edge_idx = padded_edges,
                        neighbor_mask = padded_neighbors.T, distances = padded_distances.T)
 
    def compute_paths_faster(self, node, graph):
        # thanks to tip from https://github.com/gasmichel/PathNNs_expressive/ to use igraph because it's much faster
        g = igraph.Graph.from_networkx(graph)

        lengths = []
        paths = []
        path_mask = []

        paths_s = [torch.tensor([node])]
        lengths.append(1)  # add path of length 0

        ig_paths = list(g.get_all_simple_paths(node, cutoff=self.length))
        # ig_paths_filtered = []
        # for path in ig_paths:
        #     if len(path) == self.length:
        #         ig_paths_filtered.append(path)
        #     else:
        #         do_include = True
        #         for alt_path in ig_paths:
        #             if len(alt_path) < len(path) or alt_path[0:len(path)] != path:
        #                 continue
        #             else:
        #                 do_include = False
        #                 break
                        
        #         if do_include:
        #             ig_paths_filtered.append(path)
                
        # ig_paths = ig_paths_filtered
        sp_paths = g.distances(node)[0]
        paths_s += [torch.tensor(x) for x in ig_paths]
        lengths.extend([len(x) for x in ig_paths])
        paths += paths_s
        path_mask = path_mask + [node] * len(paths_s)

        neighbors = [torch.tensor([1 if y in g.neighbors(node) else 0 for y in x]) for x in paths]
        distances = [torch.tensor([sp_paths[y] for y in x]) for x in paths]  # todo: double check what happens if graph is not connected?
        # todo: discuss what smartest way is for this
        edges = [torch.tensor([[node, -20]])]
        for idx, path in enumerate(ig_paths):
            edges += [torch.stack([torch.tensor([node, -20])] + [torch.tensor([x, y]) for x, y in zip(path[:-1], path[1:])])]
        
        return paths, path_mask, lengths, edges, neighbors, distances
