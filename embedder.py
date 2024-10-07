import torch
import torch_geometric.utils

from utils import process
import numpy as np
from torch_geometric.utils import degree, remove_self_loops
from scipy.sparse import coo_matrix
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, degree
import torch_geometric.transforms as T


class embedder:
    def __init__(self, args):
        args.gpu_num_ = args.gpu_num
        self.big_dataset = list(args.big_dataset)
        if args.gpu_num_ == -1:
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")
        if args.dataset == "dblp":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_dblp4057_mat(args.sc)
            features = process.preprocess_features(features)
        if args.dataset == "acm":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_acm_mat()
            features = process.preprocess_features(features)
        if args.dataset == "imdb":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_imdb5k_mat(args.sc)
            features = process.preprocess_features(features)
        if args.dataset == "freebase":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_freebase(args.sc)
            args.ft_size = features[0].shape[0]
            args.nb_nodes = adj_list[0].shape[0]
            args.nb_classes = labels.shape[1]
        if args.dataset in ['terrorist', 'rm', 'higgs', 'RM', 'Terrorist']:
            adj_list, features, labels = process.txt2mat(args.dataset)
            features = process.preprocess_features(features)
        if args.dataset in [
            '3sources', 'BBCSport2view_544', 'BBC4view_685', 'WikipediaArticles', 'aucs', 'citeseer', 'UCI_mfeat',
            'UCI_mfeat2', 'webKB_texas_2', 'cora', 'knn_10'
        ]:
            adj_list, features_list, labels = process.construct_dataset_through_mat_data(args.dataset)
            features_list = [process.preprocess_features(fea) for fea in features_list]

            # args.ft_size = features[0].shape[0]
            # args.nb_nodes = adj_list[0].shape[0]
            # args.nb_classes = labels.shape[1]
        if args.dataset not in [
            '3sources', 'BBCSport2view_544', 'BBC4view_685', 'WikipediaArticles', 'aucs', 'citeseer', 'UCI_mfeat',
            'UCI_mfeat2', 'webKB_texas_2', 'cora', 'knn_10'
        ]:
            args.ft_size = features.shape[1]
            features_list = []
            for i in range(args.num_view):
                features_list.append(features)
            # self.features = torch.FloatTensor(features)
            self.features = [torch.FloatTensor(features) for features in features_list]
        else:
            args.ft_size = features_list[0].shape[1]
            self.features = [torch.FloatTensor(features) for features in features_list]

        # if args.dataset in ["acm", "imdb", "freebase", "dblp", 'terrorist']:
        edge_index_list = [from_scipy_sparse_matrix(adj)[0] for adj in adj_list]
        adj_list = [process.sparse_mx_to_torch_sparse_tensor(adj) for adj in adj_list]
        if args.dataset not in self.big_dataset:
            # adj_list = [process.sparse_mx_to_torch_sparse_tensor(adj) for adj in adj_list]
            adj_list = [adj.to_dense() for adj in adj_list]


        if args.use_gdc:
            print('use graph diffusion...')
            num_nodes = self.features[0].shape[0]
            if args.gdc_k_pattern == 'eqN':
                sparsification_k = num_nodes
            transform = T.GDC(
                self_loop_weight = 1,
                normalization_in = 'sym',
                normalization_out = 'col',
                diffusion_kwargs = dict(method = 'heat', t = args.gt),  # t commonly lies in [2, 10]
                # diffusion_kwargs = dict(method = 'ppr', alpha = args.gdc_ppr_alp),
                sparsification_kwargs = dict(method = 'topk', k = sparsification_k, dim = 0),
                # sparsification_kwargs = dict(method = 'threshold', eps = args.gdc_dt),
                exact = True,
            )

            for i in range(args.num_view):
                temp = Data(edge_index = edge_index_list[i], num_nodes = num_nodes)
                temp = transform(temp)
                # degrees = degree(temp.edge_index[0], num_nodes = num_nodes)
                # degree_threshold = torch.quantile(degrees, args.gdc_dt)
                # mask = degrees[temp.edge_index[0]] > degree_threshold
                # if sum(mask) > temp.edge_index.size()[1] * 0.8:
                #     filtered_edge_index = temp.edge_index[:, mask]
                #     filtered_data = temp.edge_attr[mask]
                # else:
                #     filtered_edge_index = temp.edge_index
                #     filtered_data = temp.edge_attr
                filtered_edge_index = temp.edge_index
                filtered_data = temp.edge_attr

                if args.dataset not in self.big_dataset:
                    adj_list[i] = to_dense_adj(filtered_edge_index, edge_attr = filtered_data).squeeze()
                else:
                    adj_list[i] = torch.sparse_coo_tensor(filtered_edge_index, filtered_data,
                                                                (num_nodes, num_nodes))

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

        if not args.use_gdc:
            adj_list = [process.normalize_graph(adj, args.dataset, self.big_dataset) for adj in adj_list]

        if args.sparse:
            adj_list = [adj.to_sparse() for adj in adj_list]
        args.nb_nodes = adj_list[0].shape[0]
        args.nb_classes = labels.shape[1]
        self.adj_list = adj_list
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
    args = get_args(
        model_name = "DMG",
        dataset = "acm",  # acm imdb dblp freebase
        custom_key = "Node",  # Node: node classification
        outside = True
    )
    args.dataset = 'acm'
    args.dataset = 'terrorist'
    if args.dataset in ["acm", "imdb"]:
        args.num_view = 2
    elif args.dataset in ["freebase", "dblp"]:
        args.num_view = 3
    elif args.dataset in ['terrorist']:
        args.num_view = 14
    emb = embedder(args)
    features, adj_list, labels = emb.features, emb.adj_list, emb.labels
    with open(f'{args.dataset}.txt', 'w') as fw:
        fw.write(f'{len(adj_list)} {len(features[0]) + 1} {len(features[0]) + 1}')

        for L, adj in enumerate(adj_list):
            for i in range(len(adj)):
                for j in range(len(adj[i])):
                    if adj[i][j] == 1:
                        fw.write(f"{L + 1} {i + 1} {j + 1}\n")

        print(f'{args.dataset} finished!')

    print()


