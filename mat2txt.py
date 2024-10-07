import json
import os

import torch
import numpy as np
import torch.nn as nn
import scipy.io as sio
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder
import torch as th
import torch.nn.functional as F
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset,WikiCSDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset, CoraFullDataset
from typing import Optional
from typing import Optional, Tuple, Union
import torch
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch import Tensor
import torch
from torch_geometric.utils import from_scipy_sparse_matrix
import torch
import torch_geometric.utils
from utils.process import  preprocess_features
from utils import process
import numpy as np
from torch_geometric.utils import degree, remove_self_loops
from scipy.sparse import coo_matrix
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import argparse




def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

def load_freebase(sc=3):
    type_num = 3492
    ratio = [20, 40, 60]
    # The order of node types: 0 m 1 d 2 a 3 w
    path = "data/freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    feat_m = sp.eye(type_num)
    # Because none of M, D, A or W has features, we assign one-hot encodings to all of them.
    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mwm = sp.load_npz(path + "mwm.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    # feat_m = th.FloatTensor(preprocess_features(feat_m))
    adj_list = [mam, mdm, mwm]
    adj_fusion = mam
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return adj_list, feat_m, label, train[0], val[0], test[0], adj_fusion

class embedder:
    def __init__(self, args):
        args.gpu_num_ = args.gpu_num
        if args.gpu_num_ == -1:
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")
        if args.dataset == "dblp":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_dblp4057_mat(args.sc)
        if args.dataset == "acm":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_acm_mat()
        if args.dataset == "imdb":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_imdb5k_mat(args.sc)
        if args.dataset == "freebase":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = load_freebase(args.sc)
            # features = features.todense()
            # args.ft_size = features[0].shape[0]
            # args.nb_nodes = adj_list[0].shape[0]
            # args.nb_classes = labels.shape[1]
        if args.dataset in ['terrorist', 'rm']:
            adj_list, features, labels = process.txt2mat(args.dataset)
            # args.ft_size = features[0].shape[0]
            # args.nb_nodes = adj_list[0].shape[0]
            # args.nb_classes = labels.shape[1]

        features = features.todense()
        # if args.dataset in ["acm", "imdb", "freebase", "dblp", 'terrorist']:
        edge_index_list = [from_scipy_sparse_matrix(adj)[0] for adj in adj_list]
        adj_list = [process.sparse_mx_to_torch_sparse_tensor(adj) for adj in adj_list]
        adj_list = [adj.to_dense() for adj in adj_list]
        ##############################################
        # idx_p_list = []
        sample_edge_list = []
        # for adj in adj_list:
        #     deg_list_0 = []
        #     idx_p_list_0 = []
        #     deg_list_0.append(0)
        #     A_degree = degree(adj.to_sparse()._indices()[0], features.shape[0], dtype=int).tolist()
        #     out_node = adj.to_sparse()._indices()[1]
        #     for i in range(features.shape[0]):  # features.shape[0] = nb_nodes
        #         deg_list_0.append(deg_list_0[-1] + A_degree[i])
        #     for j in range(1, args.neighbor_num+1):
        #         random_list = [deg_list_0[i] + j % A_degree[i] for i in range(features.shape[0])]
        #         idx_p_0 = out_node[random_list]
        #         idx_p_list_0.append(idx_p_0)
        #     idx_p_list.append(idx_p_list_0)
        args.nb_nodes = adj_list[0].shape[0]
        args.nb_classes = labels.shape[1]
        args.ft_size = features.shape[1]
        features_list = []
        for i in range(args.num_view):
            features_list.append(features)
        self.adj_list = adj_list
        self.features = torch.FloatTensor(features)
        self.features = [torch.FloatTensor(features) for features in features_list]
        self.labels = torch.FloatTensor(labels).to(args.device)
        # self.idx_train = torch.LongTensor(idx_train).to(args.device)
        # self.idx_val = torch.LongTensor(idx_val).to(args.device)
        # self.idx_test = torch.LongTensor(idx_test).to(args.device)
        # self.idx_p_list = idx_p_list
        self.sample_edge_list = sample_edge_list
        self.args = args
        self.edge_index_list = edge_index_list


if __name__ == '__main__':
    from main import get_args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default = 'imdb')
    parser.add_argument('--gpu_num', default = 0)
    parser.add_argument('--sc', default = 3)
    args = parser.parse_args()
    print(args.dataset)
    # args.dataset = 'terrorist'
    if args.dataset in ["acm", "imdb"]:
        args.num_view = 2
    elif args.dataset in ["freebase", "dblp"]:
        args.num_view = 3
    elif args.dataset in ['terrorist']:
        args.num_view = 14
    emb = embedder(args)
    features, adj_list, labels = emb.features, emb.adj_list, emb.labels
    with open(f'./data_converted/{args.dataset}.txt', 'w') as fw:
        fw.write(f'{len(adj_list)} {len(features[0])} {len(features[0])}\n')
        for L, adj in enumerate(adj_list):
            for i in range(len(adj)):
                for j in range(i, len(adj[i])):
                    if adj[i][j]:
                        fw.write(f"{L + 1} {i + 1} {j + 1}\n")

    features = features[0]
    with open(f'./data_converted/{args.dataset}_attributes.txt', 'w') as fw:
        fw.write(f'0 {str([0 for _ in range(len(features[0]))])}\n')
        for i in range(len(features)):
            fw.write(f'{i + 1} {str(features[i].detach().cpu().tolist())}\n')


    community = [set() for _ in range(len(labels[0]))]
    with open(f'./data_converted/{args.dataset}_communities.txt', 'w') as fw:
        for i, lbs in enumerate(labels):
            for j, b in enumerate(lbs):
                if b:
                    community[j].add(i + 1)
                    break
        for cmu in community:
            cmu = ' '.join([str(t) for t in list(cmu)])
            fw.write(f'{cmu}\n')

    print(f'{args.dataset} finished!')




    print()


