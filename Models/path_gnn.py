"""
Somewhat inspired by https://github.com/gasmichel/PathNNs_expressive/blob/main/benchmarks/model_ogb.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Sequential, Linear, ReLU, ModuleList
from torch_geometric.nn import aggr
from torch_geometric.nn.models import JumpingKnowledge

from Models.path_conv import PathConv
from Models.path_gnn_utils import PathSequential


class PathGNN(torch.nn.Module):
    def __init__(self, emb_dim, lstm_depth, num_classes, num_tasks, nr_lstms, num_mlp_layers, node_encoder, edge_encoder, path_length, activation, JK,
                 readout_agg='sum', path_agg='sum', drop_out=0, share_lstm=False, mark_neighbors=True, shortest_path_emb=True):
        """
        :param lstm_in: size of initial features
        :param lstm_out: hidden dimension of path embeddings
        :param lstm_layers: number of layers in LSTM
        :param path_layers: number of path convolution layers
        :param mlp_layers: number of layers in MLP used for classification/regression
        :param num_out: number of nodes in final MLP depending on type of task
        :param node_encoder: whether to use node encoding prior to path convolution
        :param readout_agg: type of aggregation function for readout (currently supports 'mean' and 'sum')
        :param path_agg: type of aggregation function for path embeddings
        """
        super().__init__()

        self.drop_out = torch.nn.Dropout(p = drop_out)
        self.path_agg = path_agg
        self.encoder = None
        self.activation = activation
        self.num_classes =  num_classes
        self.num_tasks = num_tasks
        self.num_mlp_layers = num_mlp_layers
        self.emb_dim = emb_dim
        self.path_length = path_length
        self.nr_lstms = nr_lstms
        self.encoder = node_encoder 
        self.edge_encoder = edge_encoder
        self.mark_neighbors = mark_neighbors

        self.path_conv = PathSequential()
        self.path_convs = torch.nn.ModuleList([])
        self.batch_norms = torch.nn.ModuleList([])
        self.mlps = torch.nn.ModuleList([])
        
        if JK == "concat":
            self.JK = JumpingKnowledge("cat")
        elif JK == "mean":
            self.JK = lambda ls: torch.squeeze(torch.mean(torch.stack(ls), 0, True), 0)
        elif JK == "last":
            self.JK = lambda ls: ls[-1]
            
        self.shortest_path_emb = shortest_path_emb
        if self.shortest_path_emb:
            # 0...original node
            # path_length + 1: padding value 
            self.distance_embedding =  torch.nn.Embedding(path_length+2, emb_dim)
        else:
            self.distance_embedding = torch.nn.Parameter(torch.tensor(torch.ones(path_length+1, 1, emb_dim), requires_grad=True))
            torch.nn.init.xavier_normal_(self.distance_embedding)
        
        neighbor_enc_size = 1 if self.mark_neighbors else 0
        
        if share_lstm:
            self.lstm = nn.LSTM(3*emb_dim + neighbor_enc_size, emb_dim, lstm_depth, batch_first=True)
        
        for _ in range(nr_lstms):  # dynamically add path layers 
            if share_lstm:
                print("Sharing LSTM")
                lstm = self.lstm
            else:
                lstm = nn.LSTM(3*emb_dim + neighbor_enc_size, emb_dim, lstm_depth, batch_first=True)
                
            self.path_convs.append(PathConv(lstm, path_length=path_length, emb_dim=emb_dim, drop_out=self.drop_out, use_edge_feats=True, mark_neighbors=self.mark_neighbors))
            self.batch_norms.append(BatchNorm1d(emb_dim))
            self.mlps.append(Sequential(
                Linear(emb_dim, emb_dim),
                BatchNorm1d(emb_dim),
                activation,
                Linear(emb_dim, emb_dim),
                BatchNorm1d(emb_dim),
                activation,
            ))
        
        if readout_agg == 'sum':
            self.readout = aggr.SumAggregation()
        else:
            self.readout = aggr.MeanAggregation()

        # Final layer MLP
        mlp = ModuleList([])
        mlp_input_dim = emb_dim*nr_lstms if JK == "concat" else emb_dim
        for i in range(self.num_mlp_layers):
            in_size = emb_dim if i > 0 else mlp_input_dim
            out_size = emb_dim if i < self.num_mlp_layers - 1 else self.num_classes*self.num_tasks

            new_linear_layer = Linear(in_size, out_size)

            mlp.append(new_linear_layer)

            if self.num_mlp_layers > 0 and i < self.num_mlp_layers - 1:
                mlp.append(self.drop_out)
                mlp.append(activation)
                
        self.mlp = mlp
        
    def compute_embedding(self, data):
        path_index, mask_index, path_lengths, x, batch, distances = data.path_index, data.mask_index, data.path_lengths, data.x, data.batch, data.distances
        distances = distances.T

        if self.mark_neighbors:
            neighbor_mask = data.neighbor_mask.T.unsqueeze(-1)
        else:
            neighbor_mask = None
                
        if self.edge_encoder:
            path_edge_idx, edge_attr = data.path_edge_idx, data.edge_attr
            if edge_attr is not None and len(edge_attr.shape) == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.edge_encoder(edge_attr)
            new_edge_attr = torch.cat((edge_attr, torch.zeros((1, edge_attr[1].shape[0])).to(edge_attr.device)), dim=0)  # add extra entry for zeros for path of length 0
            path_edge_idx = torch.where(path_edge_idx != -20, path_edge_idx, new_edge_attr.shape[0] - 1)
            edge_attr = new_edge_attr[path_edge_idx[:, :]]

        if self.shortest_path_emb:
            distance_emb_expanded = self.distance_embedding(distances)       
        else:
            distance_emb_expanded = self.distance_embedding.expand(-1, path_index.shape[1], -1)

        x = self.encoder(x)        
        ls = []
        for i in range(self.nr_lstms):

            x2 = self.path_convs[i](path_index, mask_index, path_lengths, self.path_agg, x, edge_attr, neighbor_mask, distance_emb_expanded)
            
            # Residual connection
            x = self.batch_norms[i](x2 + x) 
            x = self.drop_out(x)
            x = self.mlps[i](x)
            ls.append(x)
            
        x = self.JK(ls)

        return self.readout(x, batch)

    def forward(self, data):
        h_graph = self.compute_embedding(data)

        for layer in self.mlp:
            h_graph = layer(h_graph)

        if self.num_tasks == 1:
            h_graph = h_graph.view(-1, self.num_classes)
        else:
            h_graph.view(-1, self.num_tasks, self.num_classes)
        return h_graph



class PathGNNSynthetic(torch.nn.Module):
    def __init__(self, emb_dim, lstm_depth, num_classes, num_tasks, nr_lstms, num_mlp_layers, node_encoder, edge_encoder, path_length, activation,
                 readout_agg='sum', path_agg='sum', drop_out=0.5, share_lstm=False, mark_neighbors=True, shortest_path_emb=True):
        """
        :param lstm_in: size of initial features
        :param lstm_out: hidden dimension of path embeddings
        :param lstm_layers: number of layers in LSTM
        :param path_layers: number of path convolution layers
        :param mlp_layers: number of layers in MLP used for classification/regression
        :param num_out: number of nodes in final MLP depending on type of task
        :param node_encoder: whether to use node encoding prior to path convolution
        :param readout_agg: type of aggregation function for readout (currently supports 'mean' and 'sum')
        :param path_agg: type of aggregation function for path embeddings
        """
        super().__init__()

        if drop_out == 0:
            self.drop_out = torch.nn.Identity()
        else:
            self.drop_out = torch.nn.Dropout(p = drop_out)
            
        self.path_agg = path_agg
        self.encoder = None
        self.activation = activation
        self.num_classes =  num_classes
        self.num_tasks = num_tasks
        self.num_mlp_layers = num_mlp_layers
        self.emb_dim = emb_dim
        self.path_length = path_length
        self.nr_lstms = nr_lstms
        self.encoder = node_encoder 
        self.edge_encoder = edge_encoder
        self.share_lstm = share_lstm
        self.mark_neighbors = mark_neighbors

        self.path_conv = PathSequential()
        self.path_convs = torch.nn.ModuleList([])
        self.batch_norms = torch.nn.ModuleList([])
        self.mlps = torch.nn.ModuleList([])

        self.shortest_path_emb = shortest_path_emb
        if self.shortest_path_emb:
            # 0...original node
            # path_length + 1: padding value 
            self.distance_embedding =  torch.nn.Embedding(path_length+2, emb_dim)
        else:
            self.distance_embedding = torch.nn.Parameter(torch.tensor(torch.ones(path_length+1, 1, emb_dim), requires_grad=True))
            torch.nn.init.xavier_normal_(self.distance_embedding)
        neighbor_enc_size = 1 if self.mark_neighbors else 0
        if share_lstm:
            self.lstm = nn.LSTM(2*emb_dim+ neighbor_enc_size, emb_dim, lstm_depth, batch_first=True)
            
        for _ in range(nr_lstms):  # dynamically add path layers 
            if share_lstm:
                lstm = self.lstm
            else:
                lstm = nn.LSTM(2*emb_dim + neighbor_enc_size, emb_dim, lstm_depth, batch_first=True)
            self.path_convs.append(PathConv(lstm, path_length=path_length, emb_dim=emb_dim, drop_out=self.drop_out, use_edge_feats=False, mark_neighbors=self.mark_neighbors))
            self.batch_norms.append(BatchNorm1d(emb_dim))
            
        if readout_agg == 'sum':
            self.readout = aggr.SumAggregation()
        else:
            self.readout = aggr.MeanAggregation()

        # Final layer MLP
        mlp = ModuleList([])
        for i in range(self.num_mlp_layers):
            in_size = emb_dim if i > 0 else emb_dim
            out_size = emb_dim if i < self.num_mlp_layers - 1 else self.num_classes*self.num_tasks

            new_linear_layer = Linear(in_size, out_size)

            mlp.append(new_linear_layer)

            if self.num_mlp_layers > 0 and i < self.num_mlp_layers - 1:
                mlp.append(self.drop_out)
                mlp.append(activation)
                
        self.mlp = mlp
        
    def compute_embedding(self, data):
        path_index, mask_index, path_lengths, x, batch, distances = data.path_index, data.mask_index, data.path_lengths, data.x, data.batch, data.distances
        distances = distances.T
        
        if self.mark_neighbors:
            neighbor_mask = data.neighbor_mask.T.unsqueeze(-1)
        else:
            neighbor_mask = None
        
        if self.edge_encoder:
            path_edge_idx, edge_attr = data.path_edge_idx, data.edge_attr
            if edge_attr is not None and len(edge_attr.shape) == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.edge_encoder(edge_attr)
            new_edge_attr = torch.cat((edge_attr, torch.zeros((1, edge_attr[1].shape[0])).to(edge_attr.device)), dim=0)  # add extra entry for zeros for path of length 0
            path_edge_idx = torch.where(path_edge_idx != -20, path_edge_idx, new_edge_attr.shape[0] - 1)
            edge_attr = new_edge_attr[path_edge_idx[:, :]]
            
        if self.shortest_path_emb:
            distance_emb_expanded = self.distance_embedding(distances)       
        else:
            distance_emb_expanded = self.distance_embedding.expand(-1, path_index.shape[1], -1)
            
        x = self.encoder(x)        
        for i in range(self.nr_lstms):
            x = F.normalize(x, p=2, dim=1)
            x2 = self.path_convs[i](path_index, mask_index, path_lengths, self.path_agg, x, edge_attr, neighbor_mask, distance_emb_expanded)
            # Residual connection
            x = self.batch_norms[i](x2 + x) 

        return self.readout(x, batch)
        
    def forward(self, data):
        h_graph =  self.compute_embedding(data)

        for layer in self.mlp:
            h_graph = layer(h_graph)

        if self.num_tasks == 1:
            h_graph = h_graph.view(-1, self.num_classes)
        else:
            h_graph.view(-1, self.num_tasks, self.num_classes)
        return h_graph
