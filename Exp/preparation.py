import os
import csv

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC, GNNBenchmarkDataset, GNNBenchmarkDataset
import torch.optim as optim
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import ToUndirected, Compose, OneHotDegree
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.utils.features import get_atom_feature_dims

from Models.gnn import GNN
from Models.encoder import NodeEncoder, EdgeEncoder, ZincAtomEncoder, EgoEncoder
from Models.mlp import MLP
from Models.path_gnn import PathGNN, PathGNNSynthetic
from Models.compute_paths import ComputePaths
from Misc.drop_features import DropFeatures
from Misc.add_zero_edge_attr import AddZeroEdgeAttr
from Misc.pad_node_attr import PadNodeAttr

exp_datasets = ["exp", "sr25"]
exp_deg = 20

def get_transform(args, split = None):
    transforms = []
    if args.dataset.lower() == "csl":
        transforms.append(OneHotDegree(5))
    elif args.dataset.lower() in exp_datasets:
        transforms.append(OneHotDegree(exp_deg))

    if "pathGNN" in args.model:
        print("Adding path pre-computation.")
        transforms.append(ComputePaths(length=args.path_length, fast=True, reverse=args.do_reverse))
        
    # Pad features if necessary (needs to be done after adding additional features from other transformation)
    if args.dataset.lower() == "csl" or args.dataset.lower() in exp_datasets:
        transforms.append(AddZeroEdgeAttr(args.emb_dim))
      
    if args.do_drop_feat:
        transforms.append(DropFeatures(args.emb_dim))

    return Compose(transforms)

def load_dataset(args, config):
    transform = get_transform(args)

    if transform is None:
        dir = os.path.join(config.DATA_PATH, args.dataset, "Original")
    else:
        print(repr(transform))
        trafo_str = repr(transform).replace("\n", "")
        dir = os.path.join(config.DATA_PATH, args.dataset, trafo_str)

    if args.dataset.lower() == "zinc":
        datasets = [ZINC(root=dir, subset=True, split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    elif args.dataset.lower() == "cifar10":
        datasets = [GNNBenchmarkDataset(name ="CIFAR10", root=dir, split=split, pre_transform=Compose([ToUndirected(), transform])) for split in ["train", "val", "test"]]
    elif args.dataset.lower() == "cluster":
        datasets = [GNNBenchmarkDataset(name ="CLUSTER", root=dir, split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    elif args.dataset.lower() in ["ogbg-molhiv", "ogbg-ppa", "ogbg-code2", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast", "ogbg-molfreesolv", "ogbg-mollipo"]:
        dataset = PygGraphPropPredDataset(root=dir, name=args.dataset.lower(), pre_transform=transform)
        split_idx = dataset.get_idx_split()
        datasets = [dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]]]
    elif args.dataset.lower() == "csl":
        all_idx = {}
        for section in ['train', 'val', 'test']:
            with open(os.path.join(config.SPLITS_PATH, "CSL",  f"{section}.index"), 'r') as f:
                reader = csv.reader(f)
                all_idx[section] = [list(map(int, idx)) for idx in reader]
        dataset = GNNBenchmarkDataset(name ="CSL", root=dir, pre_transform=transform)
        datasets = [dataset[all_idx["train"][args.split]], dataset[all_idx["val"][args.split]], dataset[all_idx["test"][args.split]]]
    else:
        raise NotImplementedError("Unknown dataset")
        
    train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(datasets[1], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(datasets[2], batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_model(args, num_classes, num_vertex_features, num_tasks):
    node_feature_dims = []
    
    model = args.model.lower()
    
    
    node_encoder_emb_dim = args.emb_dim
    # # Factor to reduce emb_dim of node_encoders for marking neighbors in pathGNNs
    # if args.do_mark_neighbors and args.model in ["pathGNN","pathGNN-S"]:
    #     node_encoder_emb_dim -= 1

    if args.dataset.lower() == "zinc"and not args.do_drop_feat:
        node_feature_dims.append(21)
        node_encoder = NodeEncoder(emb_dim=node_encoder_emb_dim, feature_dims=node_feature_dims)
        edge_encoder =  EdgeEncoder(emb_dim=args.emb_dim, feature_dims=[4])
    elif args.dataset.lower() in ["ogbg-molhiv", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast", "ogbg-molfreesolv", "ogbg-mollipo"] and not args.do_drop_feat:

        node_feature_dims += get_atom_feature_dims()
        print("node_feature_dims: ", node_feature_dims)
        node_encoder, edge_encoder = NodeEncoder(node_encoder_emb_dim, feature_dims=node_feature_dims), EdgeEncoder(args.emb_dim)
    elif args.dataset == "CSL":
        node_encoder = NodeEncoder(node_encoder_emb_dim, feature_dims=[2,2,2,2,2,2])
        edge_encoder = lambda x: x
    elif args.dataset.lower() in exp_datasets:
        node_encoder = NodeEncoder(node_encoder_emb_dim, feature_dims=[2 for _ in range(exp_deg+2)])
        edge_encoder = lambda x: x
    else:
        node_encoder, edge_encoder = lambda x: x, lambda x: x
            
    if model in ["gin", "gcn", "gat"]:  
        return GNN(num_classes, num_tasks, args.num_layers, args.emb_dim, 
                gnn_type = model, virtual_node = args.use_virtual_node, drop_ratio = args.drop_out, JK = args.JK, 
                graph_pooling = args.pooling, edge_encoder=edge_encoder, node_encoder=node_encoder, 
                use_node_encoder = args.use_node_encoder, num_mlp_layers = args.num_mlp_layers)
    elif args.model.lower() == "mlp":
            return MLP(num_features=num_vertex_features, num_layers=args.num_layers, hidden=args.emb_dim, 
                    num_classes=num_classes, num_tasks=num_tasks, dropout_rate=args.drop_out, graph_pooling=args.pooling)
    elif args.model == "pathGNN":
        from torch.nn import  ReLU, GELU
        emb_dim = args.emb_dim
        # node_encoder = Sequential(Linear(1, emb_dim), BatchNorm1d(emb_dim), ReLU(),
                                #   Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU())

        if args.activation == "relu":
            activation = ReLU()
        elif args.activation == "gelu":
            activation = GELU()
        else:
            raise NotImplementedError
        return PathGNN(emb_dim = emb_dim,
                        lstm_depth= args.lstm_depth, 
                        nr_lstms=args.num_layers, 
                        num_mlp_layers=args.num_mlp_layers, 
                        readout_agg= args.pooling, 
                        share_lstm=args.do_share_lstm,
                        num_classes = num_classes,
                        num_tasks = num_tasks,
                        node_encoder = node_encoder,
                        edge_encoder = edge_encoder,
                        path_length = args.path_length,
                        activation = activation,
                        drop_out = args.drop_out,
                        mark_neighbors = args.do_mark_neighbors,
                        path_agg=args.path_pooling,
                        shortest_path_emb = args.use_shortest_path_encoding,
                        JK = args.JK)
    elif args.model == "pathGNN-S":
        from torch.nn import ReLU, GELU
        emb_dim = args.emb_dim

        if args.activation == "relu":
            activation = ReLU()
        elif args.activation == "gelu":
            activation = GELU()
        else:
            raise NotImplementedError
        return PathGNNSynthetic(emb_dim = emb_dim,
                        lstm_depth= args.lstm_depth, 
                        nr_lstms=args.num_layers, 
                        num_mlp_layers=args.num_mlp_layers, 
                        readout_agg= args.pooling, 
                        share_lstm=args.do_share_lstm,
                        num_classes = num_classes,
                        num_tasks = num_tasks,
                        node_encoder = node_encoder,
                        edge_encoder = edge_encoder,
                        path_length = args.path_length,
                        activation = activation,
                        drop_out = args.drop_out,
                        mark_neighbors = args.do_mark_neighbors,
                        path_agg=args.path_pooling,
                        shortest_path_emb = args.use_shortest_path_encoding)  
        
    else: # Probably don't need other models
        raise ValueError("Unknown model name")

    return model


def get_optimizer_scheduler(model, args, finetune = False):
    
    if finetune:
        lr = args.lr2
    else:
        lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    args.lr_scheduler_decay_steps,
                                                    gamma=args.lr_scheduler_decay_rate)
    elif args.lr_scheduler == 'None':
        scheduler = None
    elif args.lr_scheduler == "ReduceLROnPlateau":
         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                mode='min',
                                                                factor=args.lr_scheduler_decay_rate,
                                                                patience=args.lr_schedule_patience,
                                                                verbose=True)
    else:
        raise NotImplementedError(f'Scheduler {args.lr_scheduler} is not currently supported.')

    return optimizer, scheduler

def get_loss(args):
    metric_method = None
    if args.dataset.lower() == "zinc":
        loss = torch.nn.L1Loss()
        metric = "mae"
    elif args.dataset.lower() in ["ogbg-molesol", "ogbg-molfreesolv", "ogbg-mollipo"]:
        loss = torch.nn.L1Loss()
        metric = "rmse (ogb)"
        metric_method = get_evaluator(args.dataset)
    elif args.dataset.lower() in ["cifar10", "csl", "exp", "cexp"]:
        loss = torch.nn.CrossEntropyLoss()
        metric = "accuracy"
    elif args.dataset in ["ogbg-molhiv", "ogbg-moltox21", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molsider", "ogbg-moltoxcast"]:
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "rocauc (ogb)" 
        metric_method = get_evaluator(args.dataset)
    elif args.dataset == "ogbg-ppa":
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "accuracy (ogb)" 
        metric_method = get_evaluator(args.dataset)
    elif args.dataset in ["ogbg-molpcba", "ogbg-molmuv"]:
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "ap (ogb)" 
        metric_method = get_evaluator(args.dataset)
    else:
        raise NotImplementedError("No loss for this dataset")
    
    return {"loss": loss, "metric": metric, "metric_method": metric_method}

def get_evaluator(dataset):
    evaluator = Evaluator(dataset)
    eval_method = lambda y_true, y_pred: evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    return eval_method