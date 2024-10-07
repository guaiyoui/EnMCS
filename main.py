import argparse
import numpy as np
from ruamel.yaml import YAML
import os
# from models import DMG
from DMG import *
import torch
import random


def get_args(model_name, dataset, custom_key="", yaml_path=None, outside = False) -> argparse.Namespace:
    yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")
    custom_key = custom_key.split("+")[0]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=model_name)
    parser.add_argument("--custom-key", default=custom_key)
    # parser.add_argument("--dataset", default=dataset)
    # parser.add_argument("--dataset", '-d', default = 'aucs')
    parser.add_argument("--dataset", '-d', default = 'WikipediaArticles')
    parser.add_argument("--big_dataset", help = 'large dataset', default = ['higgs'])
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

    parser.add_argument("--weight_decay", default=1e-4, help="Weight decay for parameters eta", type=float)
    # parser.add_argument("--alpha", default=0.08, help="Reconstruction error coefficient", type=float)
    parser.add_argument("--batch_size", default = 4000, type = int)

    # parser.add_argument("--num_iters", default=400, help="Number of training iterations", type=int)
    # parser.add_argument("--num_iters", default=20, help="Number of training iterations", type=int)
    parser.add_argument("--inner_epochs", default=1, help="Number of inner epochs", type=int)
    parser.add_argument("--phi_num_layers", default=2, help="Number of layers for phi", type=int)
    parser.add_argument("--phi_hidden_size", default=256, help="Number of hidden neurons for phi", type=int)
    parser.add_argument("--hid_units", default=1024, help="Number of hidden neurons", type=int)
    parser.add_argument("--decolayer", default=2, help="Number of decoder layers", type=int)
    parser.add_argument("--neighbor_num", default=300, help="Number of all sampled neighbor", type=int)
    parser.add_argument("--sample_neighbor", default=30, help="Number of sampled neighbor during each iteration", type=int)
    parser.add_argument("--sample_num_divisor", default=50, help="Number of sampled edges divisor during each iteration", type=int)
    parser.add_argument("--min_sample_num", default=10, help="Number of sampled min edges during each iteration", type=int)

    parser.add_argument('--hops', type = int, default = 3, help = 'Hop of neighbors to be calculated')
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
    # wandb
    # parser.add_argument('--wandb', type = bool, default = True)
    parser.add_argument('--wandb', type = bool, default = False)
    # wandb
    # parser.add_argument('--use_gdc', action = 'store_true', help = 'Use GDC')
    parser.add_argument('--use_gdc', default = False, help = 'Use GDC')
    parser.add_argument('--gdc_alp', default = 0.35, help = '')
    parser.add_argument('--save_path', type = str, default = './checkpoint',
                        help = 'The path for the model to save')
    # parser.add_argument('--embedding_path', type = str, default = './pretrain_result/',
    #                     help = 'The path for the embedding to save')
    parser.add_argument('--export_embedding', type = int, default = 0,
                        help = 'whether to export feature embeddings')
    parser.add_argument('--save_model', type = int, default = 1,
                        help = 'whether to export feature embeddings')
    parser.add_argument('--use_pretrain', type = bool, default = False, help = 'use_pretrain')
    parser.add_argument('--nei_coef', type = float, default = 1, help = 'use_pretrain')
    parser.add_argument('--warmup_updates_prop', type = int, default = 0.1, help = 'warmup steps_prop')
    parser.add_argument('--lr_peak', type = float, default = 0.01, help = 'learning rate', choices = [0.01])
    # parser.add_argument('--end_lr', type = float, default = 0.0001, help = 'learning rate')
    parser.add_argument('--lr_coef', type = float, default = 1e-4, help = 'learning rate for cosine lr scheduler')
    
    parser.add_argument('--num_iters', type = int, default = 70)
    parser.add_argument("--beta", default = 0.4, help = "Independence constraint coefficient", type = float)
    parser.add_argument('--alpha', type = float, default = 0.8, help = 'the value to balance the match loss.')
    
    parser.add_argument('--neighbor_coef', type = float, default = 1.0, help = 'the value to balance the match loss.')
    
    parser.add_argument("--tau", default = 0.5, help = "temperature in contrastive loss", type = int)
    parser.add_argument("--gamma", default = 0.5, help = "Contrastive constraint coefficient", type = float)
    parser.add_argument("--common_name", default = '', help = "experiment name", type = str)

    if not outside:
        with open(yaml_path, encoding = 'utf-8') as args_file:
            args = parser.parse_args()
            args_key = "-".join([args.model_name, args.dataset, args.custom_key])
            try:
                parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
            except KeyError:
                raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")

    t = parser.parse_args()
    
    common_name = t.common_name
    # common_name = \
    #     'RetainU_NoComLinkLoss_CorrLossCoefBeta{}_LinkLosCoef{}_NegaLossCon'.format(
    #         t.beta, t.alpha
    #     )
    if t.use_gdc:
        common_name += '_gdc'
    parser.add_argument('--save_model_name', type = str, default = common_name, help = 'unique saved model name')
    parser.add_argument('--embedding_path', type = str, default = f'./pretrain_result/{t.dataset}/{common_name}/',
                        help = 'embedding save path')
    with open(f'./dataset_embedding_path/embedding_path_{t.dataset}.txt', 'w') as fw:
        fw.write(f'./pretrain_result/{t.dataset}/{common_name}/')
    parser.add_argument('--wandb_name', type = str, default = f'{t.dataset}_{common_name}')

    parser.add_argument('--use_lr_scheduler', type = bool, default = True)
    parser.add_argument("--lr_max", default = 1e0, help = "Learning rate for maximization", type = float)
    # original lr_min 1e-3
    parser.add_argument("--lr_min", default = 1e-2, help = "Learning rate for minimization", type = float)

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


def get_dataset_view(args):
    # args.dataset = args.dataset.lower()
    if args.dataset in ["acm", "imdb", 'BBCSport2view_544', 'WikipediaArticles', 'higgs']:
        args.num_view = 2
    elif args.dataset in ['dblp', 'freebase', '3sources', 'aucs']:
        args.num_view = 3
    elif args.dataset in ['BBC4view_685']:
        args.num_view = 4
    elif args.dataset in ['terrorist']:
        args.num_view = 14
    elif args.dataset in ['rm']:
        args.num_view = 10


def main():
    import cProfile
    setup_seed(0)
    args = get_args(
        model_name="DMG",
        dataset="terrorist",  # acm imdb dblp freebase terrorist
        custom_key="Node",  # Node: node classification
    )
    with open('args.yaml', encoding = 'utf-8') as yaml_file:
        config = dict(YAML().load(yaml_file)['DMG-imdb-Node'])

    get_dataset_view(args)
    printConfig(args)
    gdc_alp_list = [i * 0.05 for i in range(1, 10)]
    gdc_alp_list = [0.35]
    # def run(embedder):
    #     embedder.training()
    for alp in gdc_alp_list:
        args.gdc_alp = alp
        embedder = DMG(args)

        embedder.training()
        # cProfile.run('run')
        # common, private = embedder.training()
    # return macro_f1s, micro_f1s


if __name__ == '__main__':
    'HTTP_PROXY=http://127.0.0.1:7890;HTTPS_PROXY=http://127.0.0.1:7890'
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    # macro_f1s, micro_f1s = main()
    main()

    # from tkinter import messagebox
    # import tkinter as tk
    #
    # root = tk.Tk()
    # root.withdraw()
    # window = tk.Toplevel(root)
    # window.attributes('-topmost', True)
    # messagebox.showinfo('提示', 'multilayer graph search 运行完成!')
    # window.destroy()
    '''
    pattern: both_std_scores
higgs
100%|██████████| 150/150 [00:19<00:00,  7.78it/s]
100%|██████████| 150/150 [00:14<00:00, 10.23it/s]
start EM...
max_iter: 100
generate data: 890.3486449718475 s
EM time: 34574.36079835892 s
EM iter out num: 0
recall:  tensor(0.9964) pre:  tensor(0.9667)
common / private: 1 / -2.0
F1 score by maximum weight gain (global search): 0.9813
higgs
100%|██████████| 150/150 [00:07<00:00, 20.03it/s]
100%|██████████| 150/150 [00:05<00:00, 27.38it/s]
start EM...
max_iter: 100
generate data: 1754.4019083976746 s
    '''
