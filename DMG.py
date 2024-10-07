import os
import time

import torch.nn as nn
from evaluate import evaluate
from embedder import embedder
from utils.process import GCN, update_S, drop_feature, Linearlayer
import numpy as np
from tqdm import tqdm
import random as random
import torch
from typing import Any, Optional, Tuple
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, degree
from model import Transgraph, GNNDAE, Measure_F
import wandb
from datetime import datetime
from functions import trainmultiplex
from lr import CosineDecayLR
# from functions import coo_matrix_to_nx_graph_efficient,
from functions import *
import torch.utils.data as Data
import scipy.sparse as sp


class DMG(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.criteria = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        self.log_sigmoid = nn.LogSigmoid()
        # ***
        self.feature_mlp = nn.ModuleList()
        for _ in range(args.num_view):
            self.feature_mlp.append(nn.Linear(args.ft_size, args.ft_size).to(self.args.device))
        # ***
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

        self.features = [feature.to(self.args.device) for feature in self.features]
        self.adj_list = [adj.to(self.args.device) for adj in self.adj_list]

    def invert_sparse_values_torch(self, sp_tensor):
        
        device = sp_tensor.device

        
        if not sp_tensor.is_coalesced():
            sp_tensor = sp_tensor.coalesce()

        
        indices = sp_tensor.indices()
        values = sp_tensor.values()

        inverted_values = torch.where(values == 1, torch.tensor(0, device = device), torch.tensor(1, device = device))

        inverted_sparse_tensor = torch.sparse_coo_tensor(indices, inverted_values, sp_tensor.size(),
                                                         device = device)
        return inverted_sparse_tensor



    def data_augment(self, x, adj, output_adj = False):
        # random dropout
        for i in range(self.args.num_view):
            x[i] = drop_feature(x[i], self.args.feature_drop)


        x_batch_list = []
        adj_batch_list = []
        for i in range(self.args.num_view):
            adj_batch, x_batch = transform_sp_csr_to_coo(x[i], adj[i], self.args.batch_size, x[0].shape[0], drop_last = False)
            x_batch_list.append(x_batch)
            adj_batch_list.append(adj_batch)

        def extract_and_remove(data):
            extracted = [item.pop() for item in data]

            return data, extracted

        x_batch_list, x_last_item = extract_and_remove(x_batch_list)
        adj_batch_list, adj_last_item = extract_and_remove(adj_batch_list)

        # print(x_batch_list)
        # print(len(x_batch_list), len(x_batch_list[0]), len(x_batch_list[0][0]))
        # print(len(x_last_item), x_last_item[0].shape)
        if x_batch_list[0]:
            print(len(x_batch_list), len(x_batch_list[0]), len(x_batch_list[0][0]), len(x_batch_list[0][0][0]))
            # x_batch_list = torch.from_numpy(np.array(x_batch_list).transpose(1, 0, 2, 3))
            # adj_batch_list = torch.from_numpy(np.array(adj_batch_list).transpose(1, 0, 2, 3))
            # x_last_item = torch.stack(x_last_item)
            # adj_last_item = torch.stack(adj_last_item)
            
            for i in range(len(x_batch_list[0])):
                print(i, len(x_batch_list[0][i]), len(x_batch_list[0][i][0]))
                for j in range(len(x_batch_list[0][i])):
                    if len(x_batch_list[0][i][j]) != 64:
                        print("xxxxxxxxxxxxxxxxxxxxxxxxxx")
                        print(j, len(x_batch_list[0][i][j]))
            for i in range(len(x_batch_list[1])):
                print(i, len(x_batch_list[1][i]), len(x_batch_list[1][i][0]))
                for j in range(len(x_batch_list[1][i])):
                    if len(x_batch_list[1][i][j]) != 64:
                        print("xxxxxxxxxxxxxxxxxxxxxxxxxx")
                        print(j, len(x_batch_list[1][i][j]))
            
            # x_batch_list = [sublist[:len(x_batch_list[0])-1] for sublist in x_batch_list]
            # adj_batch_list = [sublist[:len(adj_batch_list[0])-1] for sublist in adj_batch_list]
            # x_batch_list = torch.from_numpy(np.array(x_batch_list).transpose(1, 0, 2, 3))
            # adj_batch_list = torch.from_numpy(np.array(adj_batch_list).transpose(1, 0, 2, 3))
            # x_last_item = [sublist[len(x_batch_list[0])] for sublist in x_batch_list]
            # adj_last_item = [sublist[len(adj_batch_list[0])] for sublist in adj_batch_list]
            # x_last_item = torch.stack(x_last_item)
            # adj_last_item = torch.stack(adj_last_item)

            x_batch_list = torch.stack([torch.stack(x) for x in x_batch_list])
            adj_batch_list = torch.stack([torch.stack(x) for x in adj_batch_list])
            x_batch_list = x_batch_list.permute(1, 0, 2, 3)
            adj_batch_list = adj_batch_list.permute(1, 0, 2, 3)

            # x_last_item = torch.stack([torch.stack(x) for x in x_last_item])
            # adj_last_item = torch.stack([torch.stack(x) for x in adj_last_item])
            # x_last_item = x_last_item.permute(1, 0, 2, 3)
            # adj_last_item = adj_last_item.permute(1, 0, 2, 3)
            x_last_item = torch.stack(x_last_item)
            adj_last_item = torch.stack(adj_last_item)

            
            
        else:
            x_batch_list = [torch.stack(x_last_item)]
            adj_batch_list = [torch.stack(adj_last_item)]
            x_last_item = None
            adj_last_item = None

        negation_adj = []
        for i in range(len(adj)):
            if self.args.dataset in ['higgs']:
                neta = self.invert_sparse_values_torch(adj[i])
            else:
                neta = torch.ones(adj[i].shape).cuda(adj[i].device) - adj[i]
            negation_adj.append(neta)

        return x_batch_list, adj_batch_list, torch.stack(negation_adj).to(self.args.device), x_last_item, adj_last_item

    def training(self, return_embedd = False):
        seed = self.args.seed

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # # ===================================================#

        # features = [feature.to(self.args.device) for feature in self.features]
        # adj_list = [adj.to(self.args.device) for adj in self.adj_list]
        edge_index_list = [edge_index.to(self.args.device) for edge_index in self.edge_index_list]
        # wyh add_gdc 24-3-12
        # features = [self.feature_mlp[i](feature.to(self.args.device)) for i, feature in enumerate(self.features)]
        # if self.args.use_gdc:
        #     # for i in range(self.args.num_view):
        #     # transform = T.GDC(
        #     #     self_loop_weight = 1,
        #     #     normalization_in = 'sym',
        #     #     normalization_out = 'col',
        #     #     # diffusion_kwargs = dict(method = 'ppr', alpha = 0.05),
        #     #     diffusion_kwargs = dict(method = 'ppr', alpha = 0.15),
        #     #     # sparsification_kwargs = dict(method = 'topk', k = 128, dim = 0),
        #     #     exact = True,
        #     # )
        #     transform = T.GDC(
        #         self_loop_weight = 1,
        #         normalization_in = 'sym',
        #         normalization_out = 'col',
        #         method = 'ppr',
        #         eps = 1e-4,
        #         threshold = 0.01,
        #         # diffusion_kwargs = dict(method = 'ppr', alpha = 0.05),
        #         # diffusion_kwargs = dict(method = 'ppr', alpha = 0.15),
        #         # sparsification_kwargs = dict(method = 'topk', k = 128, dim = 0),
        #         exact = True,
        #     )
        #     for i in range(self.args.num_view):
        #         temp = Data(x = self.features[i], edge_index = edge_index_list[i])
        #         temp = transform(temp)
        #         self.features[i] = temp.x
        #         self.adj_list[i] = to_dense_adj(temp.edge_index, edge_attr = temp.edge_attr).squeeze()

        print('start generating each view\'s graph structure...')
        sg = time.time()
        # save each view adjacence
        import pickle
        if not os.path.exists(f'{self.args.embedding_path}'):
            os.makedirs(f'{self.args.embedding_path}')
        for i in range(self.args.num_view):
            if os.path.exists('{}/{}_g{}.pickle'.format(self.args.embedding_path, self.args.dataset, i)):
                continue
            graph = coo_matrix_to_nx_graph_efficient(self.adj_list[i])
            with open(
                    '{}/{}_g{}.pickle'.format(self.args.embedding_path, self.args.dataset, i), 'wb'
            ) as fw:
                pickle.dump(graph, fw)
        eg = time.time()
        print('end generating each view\'s graph structure, using {} s'.format(eg - sg))

        # features, adj_list, self.negation_adj = self.data_augment(self.features, self.adj_list)
        x_batch_list, adj_batch_list, self.negation_adj, x_last_item, adj_last_item \
            = self.data_augment(self.features, self.adj_list)


        # process big dataset


        print("Started training...")
        start_training = time.time()
        ae_model = GNNDAE(self.args).to(self.args.device)
        # graph independence regularization network
        mea_func = []
        for i in range(self.args.num_view):
            mea_func.append(Measure_F(self.args.c_dim, self.args.p_dim,
                                  [self.args.phi_hidden_size] * self.args.phi_num_layers,
                                  [self.args.phi_hidden_size] * self.args.phi_num_layers).to(self.args.device))
        # Optimizer
        if self.args.num_view == 2:
            optimizer = torch.optim.Adam([
                {'params': mea_func[0].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
                {'params': mea_func[1].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
                {'params': ae_model.parameters(), 'lr': self.args.lr_min}
            ], lr=self.args.lr_min)
        else:
            optimizer_list = [
                {'params': mea_func[i].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay} \
                for i in range(self.args.num_view)
            ] + [{'params': ae_model.parameters(), 'lr': self.args.lr_min}]
            optimizer = torch.optim.Adam(optimizer_list, lr = self.args.lr_min)
            # optimizer = torch.optim.Adam([
            #     {'params': mea_func[0].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
            #     {'params': mea_func[1].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
            #     {'params': mea_func[2].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
            #     {'params': ae_model.parameters(), 'lr': self.args.lr_min}
            # ], lr=self.args.lr_min)
        if self.args.use_lr_scheduler:
            lr_scheduler = CosineDecayLR(
                optimizer,
                warmup_updates_prop = self.args.warmup_updates_prop,
                tot_updates = self.args.num_iters * len(x_batch_list),
                lr = self.args.lr_min,
                lrf = self.args.lr_coef,
                mea_func_lr = self.args.lr_max,
                view = self.args.num_view
            )
        # model.train()
        ae_model.train()
        mea_func[0].train()
        mea_func[1].train()
        if self.args.num_view == 3:
            mea_func[2].train()
        best = 1e9
        cnt_wait = 0

        if self.args.use_pretrain:
            ae_model.load_state_dict(
                # torch.load('checkpoint/best_{}_{}.pt'.format(self.args.dataset, self.args.custom_key))
                torch.load(self.args.pretrain_model_path)
            )
        else:
            if self.args.wandb:
                config = dict(
                    epochs = self.args.num_iters,
                    # lr = self.args.lr,
                    # optimizer = self.args.optimizer,
                )
                if self.args.use_gdc:
                    name = self.args.wandb_name + '_' + datetime.now().strftime('%m-%d_%H-%M') + '_gdc'
                else:
                    # name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:]
                    name = self.args.wandb_name + '_' + datetime.now().strftime('%m-%d_%H-%M')
                wandb.init(config = config,
                           project = 'unsupervised_multilayer_graph',
                           entity = 'xdu_ai',
                           # name = f'{self.args.model_name}_{self.args.method_name}_{self.args.melt}',
                           name = name,
                           )
            # ***
            # features_copy = [f.clone().detach() for f in features]
            # adj_list_copy = [adj.clone().detach() for adj in adj_list]
            # self.negation_adj_copy = [_adj.clone().detach() for _adj in self.negation_adj]
            # ***

            loss_all = []
            early_stop_flag = False
            model_save_path = 'best_{}_{}_new.pt'.format(self.args.dataset, self.args.custom_key)
            first_change_model_save_path = True
            for itr in tqdm(range(1, self.args.num_iters + 1)):
                for x_batch, adj_batch in zip(x_batch_list, adj_batch_list):
                    features = x_batch.to(self.args.device)
                    adj_list = adj_batch.to(self.args.device)
                    # features = [torch.tensor(t).to(self.args.device) for t in x_batch]
                    # adj_list = [torch.tensor(t).to(self.args.device) for t in adj_batch]
                # Solve the S subproblem
                    U = ae_model.get_common(features, adj_list)  # 删除 SVD
                    # U = update_S(ae_model, features, adj_list, self.args.c_dim, self.args.device)

                    # Update network for multiple epochs
                    if self.args.inner_epochs > 1:
                        print()

                    for innerepoch in range(self.args.inner_epochs):
                        # ***
                        # loss = trainmultiplex(ae_model, mea_func, U_copy, features_copy, adj_list_copy, self.negation_adj_copy,
                        #                       self.args, optimizer, self.args.device, itr * innerepoch)
                        # ***
                        # Backprop to update
                        loss = trainmultiplex(ae_model, mea_func, U, features, adj_list, self.negation_adj,
                                              self.args, optimizer, self.args.device, itr*innerepoch)
                        loss_all.append(loss.item())
                        if self.args.use_lr_scheduler:
                            lr_scheduler.step()
                        # print('now lr:', optimizer.param_groups[2]['lr'])
                        now_lr = [group['lr'] for group in optimizer.param_groups]
                        print('now lr:', now_lr)
                        if self.args.inner_epochs > 1:
                            print('inner loss:', loss.item())
                    if loss < best:
                        best = loss
                        cnt_wait = 0
                        if self.args.save_model:
                            path = self.args.save_path
                            if not os.path.exists(path):
                                os.makedirs(path)

                            drop_duplicate_id = 1
                            while os.path.exists(os.path.join(path, model_save_path)) and first_change_model_save_path:
                                model_save_path = 'best_{}_{}_new_{}.pt'.format(
                                    self.args.dataset, self.args.custom_key, drop_duplicate_id)
                                drop_duplicate_id += 1
                            first_change_model_save_path = False
                            torch.save(ae_model.state_dict(), os.path.join(path, model_save_path))
                    elif loss > best and itr > 100:
                        cnt_wait += 1
                    if cnt_wait == self.args.patience:
                        print("Early stopped!")
                        early_stop_flag = True
                        # os.rename(
                        #     os.path.join(path, model_save_path),
                        #     os.path.join(path, '{}_{}_loss_{}.pt'.format(self.args.dataset, self.args.custom_key, best))
                        # )
                        break
                    if self.args.wandb:
                        wandb.log(dict(train_loss = loss))
                    print('====> Iteration: {} Loss = {:.4f}'.format(itr, loss))

                if x_last_item is None:
                    continue
                features = x_last_item.to(self.args.device)
                adj_list = adj_last_item.to(self.args.device)
                U = ae_model.get_common(features, adj_list)  # 删除 SVD
                # U = update_S(ae_model, features, adj_list, self.args.c_dim, self.args.device)

                # Update network for multiple epochs
                if self.args.inner_epochs > 1:
                    print()

                for innerepoch in range(self.args.inner_epochs):
                    # ***
                    # loss = trainmultiplex(ae_model, mea_func, U_copy, features_copy, adj_list_copy, self.negation_adj_copy,
                    #                       self.args, optimizer, self.args.device, itr * innerepoch)
                    # ***
                    # Backprop to update
                    loss = trainmultiplex(ae_model, mea_func, U, features, adj_list, self.negation_adj,
                                          self.args, optimizer, self.args.device, itr * innerepoch)
                    if self.args.use_lr_scheduler:
                        lr_scheduler.step()
                    # print('now lr:', optimizer.param_groups[2]['lr'])
                    now_lr = [group['lr'] for group in optimizer.param_groups]
                    print('now lr:', now_lr)
                    if self.args.inner_epochs > 1:
                        print('inner loss:', loss.item())
                if loss < best:
                    best = loss
                    cnt_wait = 0
                    if self.args.save_model:
                        path = self.args.save_path
                        if not os.path.exists(path):
                            os.makedirs(path)

                        drop_duplicate_id = 1
                        while os.path.exists(os.path.join(path, model_save_path)) and first_change_model_save_path:
                            model_save_path = 'best_{}_{}_new_{}.pt'.format(
                                self.args.dataset, self.args.custom_key, drop_duplicate_id)
                            drop_duplicate_id += 1
                        first_change_model_save_path = False
                        torch.save(ae_model.state_dict(), os.path.join(path, model_save_path))
                elif loss > best and itr > 100:
                    cnt_wait += 1
                if cnt_wait == self.args.patience:
                    print("Early stopped!")
                    early_stop_flag = True
                    # os.rename(
                    #     os.path.join(path, model_save_path),
                    #     os.path.join(path, '{}_{}_loss_{}.pt'.format(self.args.dataset, self.args.custom_key, best))
                    # )
                    break
                if self.args.wandb:
                    wandb.log(dict(train_loss = loss))
                print('====> Iteration: {} Loss = {:.4f}'.format(itr, loss))
            if self.args.wandb:
                wandb.finish()

            if not os.path.exists(
                    os.path.join(path, '{}_{}_{}_loss_{}.pt'.format(
                        self.args.dataset, self.args.custom_key, self.args.save_model_name, best))
            ):
                os.rename(
                    os.path.join(path, model_save_path),
                    os.path.join(path, '{}_{}_{}_loss_{}.pt'.format(
                        self.args.dataset, self.args.custom_key, self.args.save_model_name, best))
                )

        # if self.args.export_embedding:
        #     self.export_embedd(ae_model, features, adj_list)

            best_model_path = os.path.join(path, '{}_{}_{}_loss_{}.pt'.format(
                    self.args.dataset, self.args.custom_key, self.args.save_model_name, best))
            ae_model.load_state_dict(torch.load(best_model_path))

        if not os.path.exists(self.args.embedding_path):
            os.makedirs(self.args.embedding_path)
        
        with open(f"./loss_results/loss_{self.args.dataset}.txt", "w") as file:
            for item in loss_all:
                file.write(f"{item}\n")
        print("start_training done, using time: {} s".format(time.time() - start_training))
        print('start exporting embedding...')
        start = time.time()
        # self.data_augment(self.features, self.adj_list, True)
        # self.export_embedd(ae_model, self.features, self.adj_list)
        self.export_embedd(ae_model, x_batch_list, adj_batch_list, x_last_item, adj_last_item)
        end = time.time()
        print('export embedding using time: {} s'.format(end - start))

        print('start evaluating...')
        # from accuracy_localsearch_no_total_common_features import run, parse_args
        # search_args = parse_args()
        # lb_range = [0, 0.3, 0.4]
        # for lb in lb_range:
        #     search_args.lammbda = lb
        #     run(search_args)


        # print("Evaluating...")
        # ae_model.eval()
        # embedding = []
        # hf = update_S(ae_model, features, adj_list, self.args.c_dim, self.args.device)
        # _, private = ae_model.embed(features, adj_list)
        # private = sum(private) / self.args.num_view
        # embedding.append(hf)
        # embedding.append(private)
        # embeddings = torch.cat(embedding, 1)
        # if return_embedd:
        #     return hf, private
        #     # return embeddings
        # macro_f1s, micro_f1s = evaluate(embeddings, self.idx_train, self.idx_val, self.idx_test, self.labels,
        #                                 task=self.args.custom_key, epoch = self.args.test_epo,lr = self.args.test_lr,
        #                                 iterater=self.args.iterater, args = self.args)  # seed=seed
        # print("Start Save Model...")

        # return macro_f1s, micro_f1s

    def export_embedd(self, best_model, features_list, adj_list, x_last, adj_last):
        if not os.path.exists(self.args.embedding_path):
            os.makedirs(self.args.embedding_path)

        best_model.eval()
        common_embedding = None
        private_embedding = None
        for item, adj in zip(features_list, adj_list):
            nodes_features = item.to(self.args.device)
            adj = adj.to(self.args.device)
            common, private = best_model.embed(nodes_features, adj)
            common = torch.stack(common).cpu().detach().numpy()
            private = torch.stack(private).cpu().detach().numpy()

            if common_embedding is None and private_embedding is None:
                common_embedding, private_embedding = common, private
            else:
                common_embedding = np.concatenate([common_embedding, common], axis = 1)
                private_embedding = np.concatenate([private_embedding, private], axis = 1)

        if x_last is not None:
            nodes_features = x_last.to(self.args.device)
            adj = adj_last.to(self.args.device)
            common, private = best_model.embed(nodes_features, adj)
            common = torch.stack(common).cpu().detach().numpy()
            private = torch.stack(private).cpu().detach().numpy()
            common_embedding = np.concatenate([common_embedding, common], axis = 1)
            private_embedding = np.concatenate([private_embedding, private], axis = 1)


        for i in range(self.args.num_view):
            # np.save('{}/{}_{}_com{}.npy'.format(
            np.save('{}/{}_com{}.npy'.format(
                # self.args.embedding_path, self.args.dataset, self.args.custom_key, i),
                self.args.embedding_path, self.args.dataset, i),
                common_embedding[i])
            # np.save('{}/{}_{}_pri{}.npy'.format(
            #     self.args.embedding_path, self.args.dataset, self.args.custom_key, i),
            np.save('{}/{}_pri{}.npy'.format(
                self.args.embedding_path, self.args.dataset, i),
                private_embedding[i])

    # 不考虑大图
    # def export_embedd(self, best_model, features_list, adj_list):
    #     if not os.path.exists(self.args.embedding_path):
    #         os.makedirs(self.args.embedding_path)
    #
    #     # best_model = GNNDAE(self.args).to(self.args.device)
    #     # best_model.load_state_dict(
    #     #     torch.load('{}/best_{}_{}_new.pt'.format(
    #     #         self.args.save_path, self.args.dataset, self.args.custom_key)))
    #
    #     best_model.eval()
    #     node_embedding = []
    #     # hf = update_S(best_model, features_list, adj_list, self.args.c_dim, self.args.device)
    #     hf = best_model.get_common(features_list, adj_list)
    #     common, private = best_model.embed(features_list, adj_list)
    #
    #     # np.save('{}/{}_{}_total_common.npy'.format(
    #     #     self.args.embedding_path, self.args.dataset, self.args.custom_key),
    #     #     hf.cpu().detach().numpy())
    #     np.save('{}/{}_total_common.npy'.format(
    #         self.args.embedding_path, self.args.dataset),
    #         hf.cpu().detach().numpy())
    #
    #     for i in range(self.args.num_view):
    #         # np.save('{}/{}_{}_com{}.npy'.format(
    #         np.save('{}/{}_com{}.npy'.format(
    #             # self.args.embedding_path, self.args.dataset, self.args.custom_key, i),
    #             self.args.embedding_path, self.args.dataset, i),
    #             common[i].cpu().detach().numpy())
    #         # np.save('{}/{}_{}_pri{}.npy'.format(
    #         #     self.args.embedding_path, self.args.dataset, self.args.custom_key, i),
    #         np.save('{}/{}_pri{}.npy'.format(
    #             self.args.embedding_path, self.args.dataset, i),
    #             private[i].cpu().detach().numpy())
    #
    #     import pickle
    #     for i in range(self.args.num_view):
    #         graph = coo_matrix_to_nx_graph_efficient(adj_list[i])
    #         with open(
    #                 '{}/{}_g{}.pickle'.format(self.args.embedding_path, self.args.dataset, i), 'wb'
    #         ) as fw:
    #             pickle.dump(graph, fw)


    # def trainer(self, features, adj_list, ae_model, mean_func, ):
    #     U = ae_model.get_common(features, adj_list)  # 删除 SVD
    #     # U = update_S(ae_model, features, adj_list, self.args.c_dim, self.args.device)
    #
    #     # Update network for multiple epochs
    #     if self.args.inner_epochs > 1:
    #         print()
    #
    #     for innerepoch in range(self.args.inner_epochs):
    #         # ***
    #         # loss = trainmultiplex(ae_model, mea_func, U_copy, features_copy, adj_list_copy, self.negation_adj_copy,
    #         #                       self.args, optimizer, self.args.device, itr * innerepoch)
    #         # ***
    #         # Backprop to update
    #         loss = trainmultiplex(ae_model, mea_func, U, features, adj_list, self.negation_adj,
    #                               self.args, optimizer, self.args.device, itr * innerepoch)
    #         if self.args.use_lr_scheduler:
    #             lr_scheduler.step()
    #         # print('now lr:', optimizer.param_groups[2]['lr'])
    #         now_lr = [group['lr'] for group in optimizer.param_groups]
    #         print('now lr:', now_lr)
    #         if self.args.inner_epochs > 1:
    #             print('inner loss:', loss.item())
    #     if loss < best:
    #         best = loss
    #         cnt_wait = 0
    #         if self.args.save_model:
    #             path = self.args.save_path
    #             if not os.path.exists(path):
    #                 os.makedirs(path)
    #
    #             drop_duplicate_id = 1
    #             while os.path.exists(os.path.join(path, model_save_path)) and first_change_model_save_path:
    #                 model_save_path = 'best_{}_{}_new_{}.pt'.format(
    #                     self.args.dataset, self.args.custom_key, drop_duplicate_id)
    #                 drop_duplicate_id += 1
    #             first_change_model_save_path = False
    #             torch.save(ae_model.state_dict(), os.path.join(path, model_save_path))
    #     elif loss > best and itr > 100:
    #         cnt_wait += 1
    #     if cnt_wait == self.args.patience:
    #         print("Early stopped!")
    #         early_stop_flag = True
    #         # os.rename(
    #         #     os.path.join(path, model_save_path),
    #         #     os.path.join(path, '{}_{}_loss_{}.pt'.format(self.args.dataset, self.args.custom_key, best))
    #         # )
    #         break
    #     if self.args.wandb:
    #         wandb.log(dict(train_loss = loss))
    #     print('====> Iteration: {} Loss = {:.4f}'.format(itr, loss))























    # def forward(self, x):
    #     node_tensor, neighbor_tensor = self.encoder(x)  # (batch_size, 1, hidden_dim), (batch_size, hops, hidden_dim)
    #     neighbor_tensor = self.readout(neighbor_tensor,
    #                                    torch.tensor([0]).to(self.config.device))  # (batch_size, 1, hidden_dim)
    #     return node_tensor.squeeze(), neighbor_tensor.squeeze()











