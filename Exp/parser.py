"""
Helper functions that do argument parsing for experiments.
"""

import argparse
import yaml
import sys
from copy import deepcopy

from Misc.config import config
from Misc.utils import transform_dict_to_args_list

def parse_args(passed_args=None):
    """
    Parse command line arguments. Allows either a config file (via "--config path/to/config.yaml")
    or for all parameters to be set directly.
    A combination of these is NOT allowed.
    Partially from: https://github.com/twitter-research/cwn/blob/main/exp/parser.py
    """

    parser = argparse.ArgumentParser(description='An experiment.')

    # Config file to load
    parser.add_argument('--config', dest='config_file', type=argparse.FileType(mode='r'),
                        help='Path to a config file that should be used for this experiment. '
                        + 'CANNOT be combined with explicit arguments')

    parser.add_argument('--tracking', type=int, default=config.use_wandb_tracking,
                        help=f'If 0 runs without tracking (Default: {str(config.use_wandb_tracking)})')


    # Parameters to be set directly
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--split', type=int, default=0,
                        help='Split for cross validation (default: 0)')
    parser.add_argument('--dataset', type=str, default="ZINC",
                            help='Dataset name (default: ZINC; other options: CSL and most datasets from ogb, see ogb documentation)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (default: 100)')
    
    parser.add_argument('--device', type=str, default="cuda:0",
                    help='Which gpu to use if any (default: 0)')
    parser.add_argument('--model', type=str, default='GIN',
                    help='Model to use (default: GIN; other options: GCN, MLP, pathGNN)')
                    
    # LR SCHEDULER
    parser.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau',
                    help='Learning rate decay scheduler (default: ReduceLROnPlateau; other options: StepLR, None; For details see PyTorch documentation)')
    parser.add_argument('--lr_scheduler_decay_rate', type=float, default=0.5,
                        help='Strength of lr decay (default: 0.5)')

    # For StepLR
    parser.add_argument('--lr_scheduler_decay_steps', type=int, default=50,
                        help='(For StepLR scheduler) number of epochs between lr decay (default: 50)')

    # For ReduceLROnPlateau
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='(For ReduceLROnPlateau scheduler) mininum learnin rate (default: 1e-5)')
    parser.add_argument('--lr_schedule_patience', type=int, default=10,
                        help='(For ReduceLROnPlateau scheduler) number of epochs without improvement until the LR will be reduced')

    parser.add_argument('--max_time', type=float, default=12,
                        help='Max time (in hours) for one run')

    parser.add_argument('--drop_out', type=float, default=0.0,
                        help='Dropout rate (default: 0.0)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Dimensionality of hidden units in models (default: 64)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='Number of message passing layers (default: 5) or number of layers of the MLP')
    parser.add_argument('--num_mlp_layers', type=int, default=1,
                        help='Number of layers in the MLP that performs predictions on the embedding computed by the GNN (default: 1)')
    parser.add_argument('--virtual_node', type=int, default=0,
                        help='Set 1 to use a virtual node, that is a node that is adjacent to every node in the graph (default: 0)')
    
    
    parser.add_argument('--JK', type=str, default="last",
                        help='Jumping knowledge (default: last; other options: concat, mean)')

    parser.add_argument('--pooling', type=str, default="mean",
                        help='Graph pooling operation to use (default: mean; other options: sum)')
    parser.add_argument('--node_encoder', type=int, default=1,
                        help="Set to 0 to disable to node encoder (default: 1)")
    parser.add_argument('--activation', type=str, default="relu",
                        help=' (default: relu; other options: gelu)')

    parser.add_argument('--drop_feat', type=int, default=0,
                        help="Set to 1 to drop all edge and vertex features from the graph (default: 0)")

    # PathGNN Parameters
                        
    parser.add_argument('--path_length', type=int, default=2,
                        help='Path length (default: 2)')
    parser.add_argument('--share_lstm', type=int, default=0,
                        help="Whether to share lstm weights across layers (default: 0)")
    parser.add_argument('--mark_neighbors', type=int, default=0,
                        help="Mark neighbors (default: 0); 0...not marking neighbors, 1...neighbor marking, 2...shortest path distance embedding + neighbor marking")
    parser.add_argument('--lstm_depth', type=int, default=2,
                        help='Number of layers per LSTM (default: 2)')
    parser.add_argument('--reverse', type=int, default=0,
                        help="Reverse paths (default: 0)")
    parser.add_argument('--path_pooling', type=str, default="mean",
                        help='Pooling operation for paths (default: mean; other options: sum)')

    # Load partial args instead of command line args (if they are given)
    if passed_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(transform_dict_to_args_list(passed_args))        

    args.__dict__["use_tracking"] = args.tracking == 1
    args.__dict__["use_virtual_node"] = args.virtual_node == 1
    args.__dict__["use_node_encoder"] = args.node_encoder == 1
    args.__dict__["do_drop_feat"] = args.drop_feat == 1
    args.__dict__["do_share_lstm"] = args.share_lstm == 1    
    args.__dict__["do_reverse"] = args.reverse == 1    
    args.__dict__["do_mark_neighbors"] = args.mark_neighbors != 0   
    args.__dict__["use_shortest_path_encoding"] = args.mark_neighbors == 2 
    
    # https://codereview.stackexchange.com/a/79015
    # If a config file is provided, write it's values into the arguments
    if args.config_file:
        data = yaml.load(args.config_file)
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for key, value in data.items():
                arg_dict[key] = value

    return args
