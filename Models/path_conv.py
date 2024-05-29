from typing import Callable, Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils import scatter
import torch.nn.functional as F

class PathConv(MessagePassing):
    def __init__(self, rnn: Callable, path_length, emb_dim, drop_out, use_edge_feats,  mark_neighbors=True, **kwargs):
        super().__init__(**kwargs)
        self.rnn = rnn
        self.path_length = path_length
        self.emb_dim = emb_dim
        self.drop_out = drop_out
        self.use_edge_feats = use_edge_feats
        self.mark_neighbors = mark_neighbors

    def reset_parameters(self):
        # super().reset_parameters()
        reset(self.rnn)

    def forward(self, path_index: Tensor, mask_index: Tensor, path_lengths: Tensor, path_agg: str, x: Tensor, edge_attr: Optional[Tensor], neighbor_mask: Optional[Tensor], distance_embedding) -> Tensor:

        path_features = x[path_index[:, :]]
        
        if self.mark_neighbors:
            path_features = torch.cat((path_features, neighbor_mask), dim=-1)
        
        if self.use_edge_feats:
            path_features = self.drop_out(torch.cat((path_features, distance_embedding, edge_attr), dim=2))
        else:
            path_features = self.drop_out(torch.cat((path_features, distance_embedding), dim=2))
        
        packed_tensor = pack_padded_sequence(path_features, path_lengths.cpu(), batch_first=False, enforce_sorted=False).float()
        _, (hidden_states, _) = self.rnn(packed_tensor)
        return scatter(hidden_states[-1], mask_index, dim=0, reduce=path_agg)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.rnn})'

