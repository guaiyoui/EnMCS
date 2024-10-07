import argparse
import time

import numpy as np
from ruamel.yaml import YAML
import os
# from models import DMG
from DMG import *
import torch
import random
from functions import *
from main import get_args as margs


def get_args(model_name, dataset, custom_key="", yaml_path=None) -> argparse.Namespace:
    yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")
    custom_key = custom_key.split("+")[0]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=model_name)
    parser.add_argument("--custom-key", default=custom_key)
    parser.add_argument("--dataset", default=dataset)
    parser.add_argument('--sc', type=float, default=3.0, help='GCN self connection')
    parser.add_argument('--sparse', type=bool, default=False, help='sparse adjacency matrix')
    parser.add_argument('--iterater', type=int, default=10, help='iterater')
    parser.add_argument('--isBias', type=bool, default=False, help='isBias')
    parser.add_argument('--activation', nargs='?', default='relu')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
    parser.add_argument('--gpu_num', type=int, default=0, help='the id of gpu to use')
    parser.add_argument('--seed', type=int, default=0, help='the seed to use')
    parser.add_argument('--test_epo', type=int, default=50, help='test_epo')
    parser.add_argument('--test_lr', type=int, default=0.3, help='test_lr')
    parser.add_argument('--dropout', type=int, default=0.1, help='dropout')
    parser.add_argument('--feature_drop', type=int, default=0.1, help='dropout of features')
    parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')
    # parser.add_argument("--c_dim", default=8, help="Dimensionality of c", type=int)
    # parser.add_argument("--p_dim", default=2, help="Dimensionality of p", type=int)
    parser.add_argument("--c_dim", default = 512, help = "Dimensionality of c", type = int)
    parser.add_argument("--p_dim", default = 512, help = "Dimensionality of p", type = int)
    parser.add_argument("--lr_max", default=1e0, help="Learning rate for maximization", type=float)
    parser.add_argument("--lr_min", default=1e-3, help="Learning rate for minimization", type=float)
    parser.add_argument("--weight_decay", default=1e-4, help="Weight decay for parameters eta", type=float)
    # parser.add_argument("--alpha", default=0.08, help="Reconstruction error coefficient", type=float)
    parser.add_argument('--alpha', type = float, default = 0.1, help = 'the value the balance the loss.')
    parser.add_argument("--beta", default=1,  help="Independence constraint coefficient", type=float)
    parser.add_argument("--lammbda", default=1, help="Contrastive constraint coefficient", type=float)
    # num_iters 参数在 args.yaml 中确定
    # parser.add_argument("--num_iters", default=400, help="Number of training iterations", type=int)
    # parser.add_argument("--num_iters", default=20, help="Number of training iterations", type=int)
    parser.add_argument("--inner_epochs", default=1, help="Number of inner epochs", type=int)
    parser.add_argument("--phi_num_layers", default=2, help="Number of layers for phi", type=int)
    parser.add_argument("--phi_hidden_size", default=256, help="Number of hidden neurons for phi", type=int)
    parser.add_argument("--hid_units", default=1024, help="Number of hidden neurons", type=int)
    parser.add_argument("--decolayer", default=2, help="Number of decoder layers", type=int)
    parser.add_argument("--neighbor_num", default=300, help="Number of all sampled neighbor", type=int)
    parser.add_argument("--sample_neighbor", default=30, help="Number of sampled neighbor during each iteration", type=int)
    parser.add_argument("--sample_num", default=50, help="Number of sampled edges during each iteration", type=int)
    parser.add_argument("--tau", default=0.5, help="temperature in contrastive loss", type=int)
    parser.add_argument('--hops', default = 3, help = 'Hop of neighbors to be calculated')
    parser.add_argument('--n_layers', type = int, default = 1,
                        help = 'Number of Transformer layers')
    parser.add_argument('--n_heads', type = int, default = 8,
                        help = 'Number of Transformer heads')
    parser.add_argument('--attention_dropout', type = float, default = 0.1,
                        help = 'Dropout in the attention layer')
    parser.add_argument('--readout', type = str, default = "mean")
    parser.add_argument('--hidden_dim', type = int, default = 512,
                        help = 'Hidden layer size')
    # model saving
    parser.add_argument('--wandb', type = bool, default = True)
    # parser.add_argument('--use_gdc', action = 'store_true', help = 'Use GDC')
    parser.add_argument('--use_gdc', default = False, help = 'Use GDC')
    parser.add_argument('--gdc_alp', default = 0.35, help = '')
    parser.add_argument('--save_path', type = str, default = './checkpoint',
                        help = 'The path for the model to save')
    parser.add_argument('--embedding_path', type = str, default = './pretrain_result/',
                        help = 'The path for the embedding to save')
    parser.add_argument('--export_embedding', type = int, default = 1,
                        help = 'whether to export feature embeddings')
    parser.add_argument('--save_model', type = int, default = 1,
                        help = 'whether to export feature embeddings')
    parser.add_argument('--use_pretrain', type = bool, default = False, help = 'use_pretrain')

    with open(yaml_path) as args_file:
        args = parser.parse_args()
        args_key = "-".join([args.model_name, args.dataset, args.custom_key])
        try:
            parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
        except KeyError:
            raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")

    # Update params from .yamls
    args = parser.parse_args()
    return args


def printConfig(args):
    arg2value = {}
    for arg in vars(args):
        arg2value[arg] = getattr(args, arg)
    print(arg2value)


def setup_seed(seed = 0):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def main():
    import cProfile
    setup_seed(0)
    args = get_args(
        model_name="DMG",
        dataset="acm",  # acm imdb dblp freebase
        custom_key="Node",  # Node: node classification
    )
    if args.dataset in ["acm", "imdb"]:
        args.num_view = 2
    else:
        args.num_view = 3

    # printConfig(args)

    embedder = DMG(args)
    ae_model = GNNDAE(args).to(args.device)
    path = r".\checkpoint\acm_Node_RetainU_NoCommonLinkLoss2_addCorrLossCoefBetaTo06_loss_2.302294969558716.pt"
    path = r".\checkpoint\acm_Node_RetainU_NoCommonLinkLoss2_addCorrLossCoefBetaTo06_loss_2.849121332168579.pt"
    path = r".\checkpoint\acm_Node_RetainU_NoCommonLinkLoss2_addCorrLossCoefBetaTo1_loss_2.152538776397705.pt"
    path = r".\checkpoint\acm_Node_RetainU_NoCommonLinkLoss_CorrLossCoefBeta04_loss_1.900335431098938.pt"
    main_args = margs('', '', outside = True)
    global embedding_path
    embedding_path = main_args.embedding_path
    args.embedding_path = embedding_path
    if not os.path.exists(args.embedding_path):
        os.makedirs(args.embedding_path)
    ae_model.load_state_dict(
        torch.load(path))
    start = time.time()
    embedder.data_augment(embedder.features, embedder.adj_list, True)
    embedder.export_embedd(ae_model, embedder.features, embedder.adj_list)
    end = time.time()
    print('export embedding using time: {} s'.format(end - start))
    # graph_list = []
    # for i in range(args.num_view):
    #     adj = torch.from_numpy(np.load(os.path.join(args.EmbeddingPath, f'{args.dataset}_view{i}.npy')))
    #     graph = coo_matrix_to_nx_graph_efficient(adj)
    #     graph_list.append(graph)
    # np.save('')


if __name__ == '__main__':
    'HTTP_PROXY=http://127.0.0.1:7890;HTTPS_PROXY=http://127.0.0.1:7890'
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    main()
