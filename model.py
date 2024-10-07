import torch
import torch.nn as nn
from layer import TransformerBlock
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GCNConv
from utils.process import GCN, update_S, drop_feature, Linearlayer
import numpy as np
from tqdm import tqdm
import random as random
import torch
from typing import Any, Optional, Tuple
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj


class Transgraph(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        self.num_view = config.num_view

        # self.Linear1 = nn.Linear(input_dim, self.config.hidden_dim)
        # self.encoder = TransformerBlock(hops = config.hops,
        #                                 input_dim = input_dim,
        #                                 n_layers = config.n_layers,
        #                                 num_heads = config.n_heads,
        #                                 hidden_dim = config.hidden_dim,
        #                                 dropout_rate = config.dropout,
        #                                 attention_dropout_rate = config.attention_dropout)
        if config.readout == "sum":
            self.readout = global_add_pool
        elif config.readout == "mean":
            self.readout = global_mean_pool
        elif config.readout == "max":
            self.readout = global_max_pool
        else:
            raise ValueError("Invalid pooling type.")

        self.marginloss = nn.MarginRankingLoss(0.5)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim * (config.hops + 1), config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim)
        )

    def forward(self, x, adj):
        # total_adj = torch.zeros_like(adj[0])
        # total_x = torch.zeros_like(x[0])
        neighbor = []
        for i in range(self.num_view):
            # total_x += x[i]
            # total_adj += adj[i]
            tmp = self.get_community_features(x[i], adj[i], K=self.config.hops)
            # print(tmp.shape)
            neighbor_tensor = self.encoder(tmp)
            # neighbor_tensor = self.readout(neighbor_tensor, torch.tensor([0]).to(self.config.device)).to(self.config.device)
            neighbor.append(neighbor_tensor.squeeze())

        # total_adj /= len(adj)
        # total_x /= len(x)
        # x.append(total_x)
        # adj.append(total_adj)
        tmp = self.get_community_features(x[-1], adj[-1], K=self.config.hops)
        neighbor_tensor = self.encoder(tmp)
        # neighbor_tensor = self.readout(neighbor_tensor, torch.tensor([0]).to(self.config.device)).to(self.config.device)
        neighbor.append(neighbor_tensor.squeeze())
        return neighbor

    # add by wyh 20240314
    def get_community_features(self, features, adj, K = 3):
        # nodes_features = torch.empty(features.shape[0], 1, K + 1, features.shape[1]).to(self.config.device)
        nodes_features = features
        # ***
        # ***
        # for i in range(features.shape[0]):
        #     nodes_features[i, 0, 0, :] = features[i]

        x = features + torch.zeros_like(features)
        # (((x == features).sum(axis = 0)) == 2708).sum()
        for i in range(K):

            x = torch.matmul(adj, x)

            nodes_features = torch.cat([nodes_features, x], dim = 1)
            # for index in range(features.shape[0]):
            #     nodes_features[index, 0, i + 1, :] = x[index]

        # nodes_features = nodes_features.squeeze()

        return nodes_features 


class GNNDAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_view = self.args.num_view
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.gcn = nn.ModuleList()

        for _ in range(self.args.num_view):
            self.encoder.append(GNNEncoder(args))
            self.decoder.append(Decoder(args))
            self.gcn.append(GCN(args.ft_size, args.hid_units, args.activation, args.dropout, args.isBias))

        self.transGraph = Transgraph(input_dim = args.ft_size, config=args)
        self.fc_common = nn.Linear(self.args.c_dim * self.args.num_view, self.args.c_dim)
        # self.hops = args.hops
        # self.input_dim = args.input_dim
        #
        # self.trans_encoder = TransformerBlock(hops = args.hops,
        #                                 input_dim = args.input_dim,
        #                                 n_layers = args.n_layers,
        #                                 num_heads = args.n_heads,
        #                                 hidden_dim = args.hidden_dim,
        #                                 dropout_rate = args.dropout,
        #                                 attention_dropout_rate = args.attention_dropout)
        # if args.readout == "sum":
        #     self.readout = global_add_pool
        # elif args.readout == "mean":
        #     self.readout = global_mean_pool
        # elif args.readout == "max":
        #     self.readout = global_max_pool
        # else:
        #     raise ValueError("Invalid pooling type.")
        #
        # self.marginloss = nn.MarginRankingLoss(0.5)

    def get_common(self, x, adj_list):
        # with torch.no_grad():
        #     common, _ = self.encode(x, adj_list, advance_forward = True)
        common, _ = self.encode(x, adj_list, advance_forward = True)
        FF = torch.cat(common, 1)
        FF = self.fc_common(FF)

        return FF

    def encode(self, x, adj_list, advance_forward = False): 
        common = []
        private = []
        if advance_forward:
            # ttmp = self.forward_gcn(x, adj_list)
            x = self.forward_gcn(x, adj_list)
        # else:
        #     ttmp = x
        for i in range(self.args.num_view):
            tmp = self.encoder[i](x[i], adj_list[i])
            common.append(tmp[0])
            private.append(tmp[1])

        return common, private

    def decode(self, s, p):
        recons = []
        for i in range(self.num_view):
            tmp = self.decoder[i](s[i], p[i])
            recons.append(tmp)

        return recons
    '''
    class GNNEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pipe = GCN(args.ft_size, args.hid_units, args.activation, args.dropout, args.isBias)
        # map to common
        self.S = nn.Linear(args.hid_units, args.c_dim)
        # map to private
        self.P = nn.Linear(args.hid_units, args.p_dim)

    def forward(self, x, adj):
        tmp = self.pipe(x, adj)
        common = self.S(tmp)
        private = self.P(tmp)
        return common, private
    '''
    def forward_gcn(self, x, adj):
        tmp = []
        for i in range(self.args.num_view):
            tmp.append(self.gcn[i](x[i], adj[i]))
        return tmp

    def forward(self, x, adj):
        x = self.forward_gcn(x, adj)
        common, private = self.encode(x, adj)
        neighbor_tensor = self.transGraph(x, adj)
        # recons = self.decode(common, private)

        return common, private, neighbor_tensor

    def embed(self, x, adj_list):
        common = []
        private = []
        x = self.forward_gcn(x, adj_list)
        for i in range(self.args.num_view):
            tmp = self.encoder[i](x[i], adj_list[i])
            common.append(tmp[0].detach())
            private.append(tmp[1].detach())
        return common, private




class GNNEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.pipe = GCN(args.ft_size, args.hid_units, args.activation, args.dropout, args.isBias)
        # map to common
        # self.S = nn.Linear(args.hid_units, args.c_dim)
        # # map to private
        # self.P = nn.Linear(args.hid_units, args.p_dim)
        self.S = nn.Linear(args.ft_size, args.c_dim)
        # map to private
        self.P = nn.Linear(args.ft_size, args.p_dim)

    def forward(self, x, adj):
        # tmp = self.pipe(x, adj)
        common = self.S(x)
        private = self.P(x)
        return common, private


class MLP(nn.Module):
    def __init__(self, input_d, structure, output_d, dropprob=0.0):
        super(MLP, self).__init__()
        self.net = nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropprob)
        struc = [input_d] + structure + [output_d]

        for i in range(len(struc)-1):
            self.net.append(nn.Linear(struc[i], struc[i+1]))

    def forward(self, x):
        for i in range(len(self.net)-1):
            x = F.relu(self.net[i](x))
            x = self.dropout(x)

        # For the last layer
        y = self.net[-1](x)

        return y

# measurable functions \phi and \psi
class Measure_F(nn.Module):
    def __init__(self, view1_dim, view2_dim, phi_size, psi_size, latent_dim=1):
        super(Measure_F, self).__init__()
        self.phi = MLP(view1_dim, phi_size, latent_dim)
        self.psi = MLP(view2_dim, psi_size, latent_dim)
        # gradient reversal layer
        # self.grl1 = GradientReversalLayer()
        # self.grl2 = GradientReversalLayer()

    def forward(self, x1, x2):
        # y1 = self.phi(grad_reverse(x1, 1))
        # y2 = self.psi(grad_reverse(x2, 1))
        y1 = self.phi(x1)
        y2 = self.psi(x2)
        return y1, y2


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.linear1 = Linearlayer(args.decolayer,args.c_dim + args.p_dim, args.hid_units, args.ft_size)
        self.linear2 = nn.Linear(args.ft_size, args.ft_size)

    def forward(self, s, p):
        recons = self.linear1(torch.cat((s, p), 1))
        recons = self.linear2(F.relu(recons))
        return recons

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

def grad_reverse(x, coeff):
    return GradientReversalLayer.apply(x, coeff)
