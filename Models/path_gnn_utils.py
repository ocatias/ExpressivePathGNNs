import torch
from ogb.utils.features import get_atom_feature_dims
from torch import nn


class PathSequential(nn.Sequential):
    def forward(self, *inputs):
        path_index, mask_index, path_lengths, path_agg = None, None, None, None

        for name, module in self._modules.items():
            if 'path' in name:
                if type(inputs) == tuple:
                    (path_index, mask_index, path_lengths, path_agg), x = inputs
                    inputs = module(path_index, mask_index, path_lengths, path_agg, x)
                else:
                    inputs = module(path_index, mask_index, path_lengths, path_agg, inputs)
            else:
                inputs = module(inputs)
        return inputs



class ZincAtomEncoder(torch.nn.Module):
    # From ESAN
    def __init__(self, policy, emb_dim):
        super(ZincAtomEncoder, self).__init__()
        self.policy = policy
        self.num_added = 2
        self.enc = torch.nn.Embedding(21, emb_dim)

    def forward(self, x):
        if self.policy == 'ego_nets_plus':
            return torch.hstack((x[:, :self.num_added], self.enc(x[:, self.num_added:].squeeze())))
        else:
            return self.enc(x.squeeze())

class NodeEncoder(torch.nn.Module):

    def __init__(self, emb_dim, feature_dims=None):
        super(NodeEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        if feature_dims is None:
            feature_dims = get_atom_feature_dims()

        for i, dim in enumerate(feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        x = x.long()
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding

class EdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, feature_dims = None):
        super(EdgeEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        if feature_dims is None:
            feature_dims = get_bond_feature_dims()

        for i, dim in enumerate(feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding   