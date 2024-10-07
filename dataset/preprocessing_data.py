import os
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
import torch
# from process import encode_onehot, preprocess_features
import json
import h5py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def run():
    # name = 'acm_view2'
    # name = 'acm_ori_view1'
    # name = 'acm_ori_view2'
    # generate_dblp_pt()
    # start('dblp')

    # generate_imdb_pt()
    # start('imdb')

    # generate_freebase_pt()
    # start('freebase')

    # 测试自己构建的 txt 文件是否正确
    # adj_list, features, labels = txt2mat('imdb')
    # original_imdb = sio.loadmat('imdb5k.mat')
    # adj1 = original_imdb['MAM']
    # adj2 = original_imdb['MDM']
    #
    # print()
    # exit()

    # 处理 txt 数据
    # name = 'terrorist'
    # name = 'rm'
    # # name = 'higgs'
    # generate_txtData_pt(name)
    # start(name)
    # exit()

    # generate_new_mat_pt('3sources', 5)
    # start('3sources')

    # 处理 new mat 数据
    name = 'BBC4view_685'
    name = 'BBCSport2view_544'
    name = 'WikipediaArticles'
    name = 'aucs'
    generate_new_mat_pt(name, 10)
    start(name)

    name = 'aucs'
    # construct_aucs()
    # generate_new_mat_pt(name, 10)
    # start(name)

    name = 'UCI_mfeat'
    name = 'webKB_texas_2'



def generate_new_mat_pt(name, k):
    path = os.path.join('../data/new', name + '.mat')
    _, _, label = construct_dataset_through_mat_data(path, k)

    label = torch.FloatTensor(label)
    # x = torch.FloatTensor(preprocess_features(x))

    y = torch.zeros(len(label))
    for i, it in enumerate(label):
        for j, t in enumerate(it):
            if t == 1:
                y[i] = j
                break
    # torch.save([torch.tensor(1), x.type(torch.LongTensor),
    torch.save([torch.tensor(1), torch.tensor(1),
                y.type(torch.LongTensor), torch.tensor(1)], f"./{name}.pt")

def construct_dataset_through_mat_data(path, k = 10):
    feature_list = []
    adj_list = []
    data = sio.loadmat(path)
    view = data['data'].shape[1]
    for i in range(view):
        temp = data['data'][0, i]
        feature_list.append(sp.lil_matrix(temp.T.astype(float)))

        df = pd.DataFrame(temp)
        corr_matrix = df.corr()

        nodes = corr_matrix.shape[0]
        adj = np.zeros((nodes, nodes), dtype = int)
        for j in range(nodes):
            indices = np.argsort(-corr_matrix.iloc[j].values)[: k + 1]
            adj[j, indices] = 1
            adj[indices, j] = 1

        adj = sp.csc_matrix(adj)
        adj_list.append(adj)

    labels = data['truelabel'][0, 0].squeeze()
    labels = encode_onehot(labels)

    return adj_list, feature_list, labels

def construct_aucs(k = 10):
    path = '../data/new/aucs.mat'
    feature_list = []
    adj_list = []
    data = sio.loadmat(path)
    view = data['data'].shape[1]
    for i in range(view):
        temp = data['data'][0, i]
        feature_list.append(sp.lil_matrix(temp.T.astype(float)))

        df = pd.DataFrame(temp)
        corr_matrix = df.corr()

        nodes = corr_matrix.shape[0]
        adj = np.zeros((nodes, nodes), dtype = int)
        for j in range(nodes):
            indices = np.argsort(-corr_matrix.iloc[j].values)[: k + 1]
            adj[j, indices] = 1
            adj[indices, j] = 1

        adj = sp.csc_matrix(adj)
        adj_list.append(adj)

    labels = data['truelabel'][0, 0].squeeze()
    labels = encode_onehot(labels)

    return adj_list, feature_list, labels



def generate_txtData_pt(name):
    _, x, label = txt2mat(name)
    label = torch.FloatTensor(label)
    # x = torch.FloatTensor(preprocess_features(x))
    x = torch.tensor(1)

    y = torch.zeros(len(label))
    for i, it in enumerate(label):
        for j, t in enumerate(it):
            if t == 1:
                y[i] = j
                break
    torch.save([torch.tensor(1), x.type(torch.LongTensor),
                y.type(torch.LongTensor), torch.tensor(1)], f"./{name}.pt")

def generate_freebase_pt():
    type_num = 3492
    ratio = [20, 40, 60]
    # The order of node types: 0 m 1 d 2 a 3 w
    path = "../data/freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    label = torch.FloatTensor(label)
    feat_m = sp.eye(type_num)
    # x = torch.FloatTensor(preprocess_features(feat_m))
    x = torch.tensor(1)

    y = torch.zeros(len(label))
    for i, it in enumerate(label):
        for j, t in enumerate(it):
            if t == 1:
                y[i] = j
                break
    torch.save([torch.tensor(1), x.type(torch.LongTensor),
                y.type(torch.LongTensor), torch.tensor(1)], f"./freebase.pt")
def generate_dblp_pt(sc = 3, retx = False):
    data = sio.loadmat('../data/DBLP4057.mat')

    # adj1 = data['MAM'] + np.eye(data['MAM'].shape[0]) * sc
    # adj2 = data["MDM"] + np.eye(data["MDM"].shape[0]) * sc
    # adj1 = sp.csr_matrix(adj1)
    # adj2 = sp.csr_matrix(adj2)
    # edge_index1 = from_scipy_sparse_matrix(adj1)[0]
    # edge_index2 = from_scipy_sparse_matrix(adj2)[0]

    lbs = data['label']
    try:
        x = torch.tensor(data['feature'])
    except KeyError:
        try:
            x = torch.tensor(data['features'])
        except KeyError:
            raise


    y = torch.zeros(len(lbs))
    for i, it in enumerate(lbs):
        for j, t in enumerate(it):
            if t == 1:
                y[i] = j
                break
    torch.save([torch.tensor(1), x.type(torch.LongTensor),
                y.type(torch.LongTensor), torch.tensor(1)], f"./dblp.pt")
    # if retx:
    #     return edge_index1, edge_index2, y, x
    # return edge_index1, edge_index2, y

def generate_imdb_pt(sc = 3, retx = False):
    data = sio.loadmat('../data/imdb5k.mat')

    adj1 = data['MAM'] + np.eye(data['MAM'].shape[0]) * sc
    adj2 = data["MDM"] + np.eye(data["MDM"].shape[0]) * sc
    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    edge_index1 = from_scipy_sparse_matrix(adj1)[0]
    edge_index2 = from_scipy_sparse_matrix(adj2)[0]

    lbs = data['label']
    x = torch.tensor(data['feature'])
    y = torch.zeros(len(lbs))
    for i, it in enumerate(lbs):
        for j, t in enumerate(it):
            if t == 1:
                y[i] = j
                break
    torch.save([torch.tensor(1), x.type(torch.LongTensor),
                y.type(torch.LongTensor), torch.tensor(1)], f"./imdb.pt")
    if retx:
        return edge_index1, edge_index2, y, x
    return edge_index1, edge_index2, y

def start(name):
    # data_list = torch.load("../cora.pt")
    # data = sio.loadmat('../data/DBLP4057.mat')

    data_list = torch.load("./{}.pt".format(name))

    labels = data_list[2]
    print(labels.shape)
    # exit()
    print(labels, torch.min(labels), torch.max(labels))
    num_class = torch.max(labels)-torch.min(labels)+1
    print(num_class)
    communities = [[i for i in range(labels.shape[0]) if labels[i] == j] for j in range(num_class)]

    print(communities, len(communities))


    selected_queries = []
    ground_truth = []

    for i in range(150):
        num_node = torch.randint(1, 4, (1,)).item()
        selected_class = torch.randint(0, num_class, (1,)).item()
        # print(num_node)
        selected_nodes = []
        for j in range(num_node):
            selected_node = torch.randint(0, len(communities[selected_class]), (1,)).item()
            selected_nodes.append(communities[selected_class][selected_node])
        selected_queries.append(selected_nodes)
        ground_truth.append(communities[selected_class])

    if not os.path.exists(name):
        os.makedirs(name)

    query_file=open(f"./{name}/{name}.query", "w")
    gt_file = open(f"./{name}/{name}.gt", "w")

    for i in range(len(selected_queries)):
        for j in range(len(selected_queries[i])):
            query_file.write(str(selected_queries[i][j]))
            query_file.write(" ")
        query_file.write("\n")
        for j in range(len(ground_truth[i])):
            gt_file.write(str(ground_truth[i][j]))
            gt_file.write(" ")
        gt_file.write("\n")

    # adj = data_list[0]
    # print(adj)
    # coalesced_tensor = adj.coalesce()
    # index = coalesced_tensor.indices()
    # print(index)

    # edge_file = open(f"./{name}/{name}.edges", "w")
    # for i in range(index.shape[1]):
    #     edge_file.write(str(index[0][i].item()))
    #     edge_file.write(" ")
    #     edge_file.write(str(index[1][i].item()))
    #     edge_file.write("\n")


def edge_index_to_sparse_coo(edge_index):
    row = edge_index[0].long()
    col = edge_index[1].long()

    # 构建稀疏矩阵的形状
    num_nodes = torch.max(edge_index) + 1
    size = (num_nodes.item(), num_nodes.item())

    # # 构建稀疏矩阵
    values = torch.ones_like(row)
    edge_index_sparse = torch.sparse_coo_tensor(torch.stack([row, col]), values, size)
    # edge_index_sparse = torch.zeros((num_nodes, num_nodes))
    # for i, j in zip(row, col):
    #     edge_index_sparse[i, j] = 1
    #     edge_index_sparse[j, i] = 1

    return edge_index_sparse


def txt2mat(data_name, sc = 3):
    dataset_file = open(os.path.join('../data', data_name + '.txt'))
    # read the first line of the file
    first_line = dataset_file.readline()
    split_first_line = first_line.split(' ')

    number_of_layers = int(split_first_line[0])
    number_of_nodes = int(split_first_line[1])
    maximum_node = int(split_first_line[2])

    # load adj
    # adj_list = [sp.csr_matrix(np.zeros((number_of_nodes, number_of_nodes))) for _ in range(number_of_layers)]
    # adj_list = [np.zeros((number_of_nodes, number_of_nodes)) for _ in range(number_of_layers)]
    adj_list = []
    for _ in range(number_of_layers):
        # 使用lil_matrix造函数直接创建空矩阵
        matrix = sp.lil_matrix((number_of_nodes, number_of_nodes), dtype = np.float32)
        adj_list.append(matrix)

    for line in dataset_file:
        split_line = line.strip().split(' ')
        layer = int(split_line[0]) - 1
        from_node = int(split_line[1]) - 1
        to_node = int(split_line[2]) - 1
        adj_list[layer][from_node, to_node] = 1
        adj_list[layer][to_node, from_node] = 1

    for i, t in enumerate(adj_list):
        adj_list[i] = sp.csr_matrix(t)

    # load features
    features = [None for _ in range(number_of_nodes)]
    dataset_features = open(os.path.join('../data', data_name + '_attributes.txt'))
    for line in dataset_features:
        split_line = line.strip().split('[')
        split_line[1] = '[' + split_line[1]
        features[int(split_line[0]) - 1] = json.loads(split_line[1])
    features = sp.lil_matrix(np.array(features).astype(float))

    # load labels
    dataset_labels = open(os.path.join('../data', data_name + '_communities.txt'))
    labels = np.zeros((number_of_nodes,))
    for i, line in enumerate(dataset_labels):
        com = line.strip().split()
        com = [int(t) - 1 for t in com]
        labels[com] = i
    labels = encode_onehot(labels)
    # set the number of layers
    # self.number_of_layers = int(split_first_line[0])
    # self.layers_iterator = range(self.number_of_layers)
    dataset_features.close()
    dataset_file.close()
    dataset_labels.close()

    return adj_list, features, labels


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


if __name__ == '__main__':
    run()



