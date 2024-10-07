import os
import torch.nn as nn
from embedder import embedder
from utils.process import GCN, update_S, drop_feature, Linearlayer
import numpy as np
from tqdm import tqdm
import random as random
import torch
from typing import Any, Optional, Tuple
import torch.nn.functional as F
from models.DMG import GNNDAE
from main import get_args
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)


class Embedding(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    @torch.no_grad
    def get_embedd(self, path):
        features = [feature.to(self.args.device) for feature in self.features]
        adj_list = [adj.to(self.args.device) for adj in self.adj_list]
        for i in range(self.args.num_view):
            features[i] = drop_feature(features[i], self.args.feature_drop)

        ae_model = GNNDAE(self.args).to(self.args.device)
        ae_model.load_state_dict(torch.load(path))
        ae_model.eval()
        hf = update_S(ae_model, features, adj_list, self.args.c_dim, self.args.device)
        _, private = ae_model.embed(features, adj_list)
        private = sum(private) / self.args.num_view
        embedding = []
        embedding.append(hf)
        embedding.append(private)
        embeddings = torch.cat(embedding, 1)

        return embeddings


def edge_index_to_sparse_coo(edge_index):
    row = edge_index[0].long()
    col = edge_index[1].long()

    num_nodes = torch.max(edge_index) + 1
    size = (num_nodes.item(), num_nodes.item())

    values = torch.ones_like(row)
    edge_index_sparse = torch.sparse_coo_tensor(torch.stack([row, col]), values, size)

    return edge_index_sparse


def get_acm_pt(x, using_DMG = True):
    from utils.process import generate_acm_pt
    name = 'acm'
    if using_DMG:
        edge_index1, edge_index2, y = generate_acm_pt()
    else:
        edge_index1, edge_index2, y, x = generate_acm_pt(retx = True)
        name = f'{name}_ori'
    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')
    torch.save([edge_index_to_sparse_coo(edge_index1).type(torch.LongTensor), x.type(torch.LongTensor),
                y.type(torch.LongTensor), edge_index1.type(torch.LongTensor)], "./dataset/" + f'{name}_view1' + "_pyg.pt")
    torch.save([edge_index_to_sparse_coo(edge_index2).type(torch.LongTensor), x.type(torch.LongTensor),
                y.type(torch.LongTensor), edge_index2.type(torch.LongTensor)], "./dataset/" + f'{name}_view2' + "_pyg.pt")


if __name__ == '__main__':
    args = get_args(
        model_name = "DMG",
        dataset = "acm",  # acm imdb dblp freebase
        custom_key = "Node",  # Node: node classification
    )
    if args.dataset in ["acm", "imdb"]:
        args.num_view = 2
    else:
        args.num_view = 3

    # e = Embedding(args)
    # x = e.get_embedd('./checkpoint/best_acm_Node.pt')
    x = None
    get_acm_pt(x, False)


