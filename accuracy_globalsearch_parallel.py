import torch
from functions import f1_score_calculation, load_query_n_gt, cosin_similarity, get_gt_legnth, evaluation
import argparse
import numpy as np
from tqdm import tqdm
from numpy import *
import time
import pickle
from EM import generate_sample_data
from EM import run as EMRun
from accelerate_EM import run as AccEMRun
import os
import cProfile
from main import get_dataset_view
import pandas as pd
import threading

def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters

    parser = argparse.ArgumentParser()
    parser.add_argument('-pt', '--pattern', type = str, default = 'both_std',
                        choices = ['com', 'pri', 'both', 'both_std', 'both_std_scores'])
    # main parameters
    parser.add_argument('--tau', type = float, default = 0.5, help = 'coef for private feature')
    parser.add_argument('--dataset', '-d', type=str, default='acm', help='dataset name')
    with open(f'./dataset_embedding_path/embedding_path_{parser.parse_args().dataset}.txt', 'r') as fr:
        embedding_path = fr.read()
    parser.add_argument('--embedding_tensor_name', type=str, help='embedding tensor name')
    parser.add_argument('--EmbeddingPath', type=str, default=embedding_path, help='embedding path')
    parser.add_argument('--topk', type=int, default=400, help='the number of nodes selected.')
    parser.add_argument('--lammbda', type = float, default = 0.5, help = 'coef for private feature')

    
    # parser.add_argument('--pattern', type = str, default = 'both_std_scores', choices = ['com', 'pri', 'both', 'both_std', 'both_std_scores'])
    # parser.add_argument('--pattern', type = str, default = 'both', choices = ['com', 'pri', 'both', 'both_std'])
    # parser.add_argument('-pt', '--pattern', type = str, default = 'both_std', choices = ['com', 'pri', 'both', 'both_std'])
    # parser.add_argument('--pattern', type = str, default = 'com', choices = ['com', 'pri', 'both', 'both_std'])
    # parser.add_argument('--pattern', type = str, default = 'pri', choices = ['com', 'pri', 'both', 'both_std'])

    parser.add_argument('--max_iter', type = int, default = 20, help = 'EM max iter')
    parser.add_argument('--write_result', type = bool, default = True)
    temp = parser.parse_args()
    parser.add_argument('--savefile', type = str, default = '_globalsearch_{}_maxIter{}.txt'.format(
        temp.pattern, temp.max_iter))

    return parser.parse_args()


def load_preprocess_data(args):
    get_dataset_view(args)

    if args.embedding_tensor_name is None:
        args.embedding_tensor_name = args.dataset

    st = time.time()
    common_embedding_tensor = []
    private_embedding_tensor = []
    for i in range(args.num_view):
        common_embedding_tensor.append(
            torch.from_numpy(np.load(
                os.path.join(args.EmbeddingPath, args.embedding_tensor_name + f'_com{i}.npy')))
            # np.load(args.EmbeddingPath + args.embedding_tensor_name + f'_com{i}.npy')
        )
        private_embedding_tensor.append(
            torch.from_numpy(np.load(
                os.path.join(args.EmbeddingPath, args.embedding_tensor_name + f'_pri{i}.npy')))
            # np.load(args.EmbeddingPath + args.embedding_tensor_name + f'_pri{i}.npy')
        )
    common_embedding_tensor = torch.stack(common_embedding_tensor)
    private_embedding_tensor = torch.stack(private_embedding_tensor)

    # load queries and labels
    print(f"the shape of common_embedding_tensor[0] is {common_embedding_tensor[0].shape}")
    query, labels = load_query_n_gt("./dataset/", args.dataset, common_embedding_tensor[0].shape[0])
    gt_length = get_gt_legnth("./dataset/", args.dataset)
    # torch.Size([150, 3025])

    graph_list = []
    for i in range(args.num_view):
        with open(os.path.join(args.EmbeddingPath, args.embedding_tensor_name + '_g{}.pickle'.format(i)), 'rb') as f:
            graph_list.append(pickle.load(f))
    et = time.time()
    print(f'load data using time: {et - st} s')

    start = time.time()
    query_num = torch.sum(query, dim=1)

    query_feature_common = []
    query_feature_private = []
    for i in range(args.num_view):
        query_feature_c = torch.mm(query, common_embedding_tensor[i])
        query_feature_common.append(torch.div(query_feature_c, query_num.view(-1, 1)))

        query_feature = torch.mm(query, private_embedding_tensor[i])
        query_feature_private.append(torch.div(query_feature, query_num.view(-1, 1)))

    query_score_common = []
    query_score_private = []
    for i in range(args.num_view):
        # query_score_c:
        # dimension (150, 3025) 
        query_score_c = cosin_similarity(query_feature_common[i], common_embedding_tensor[i])  # (query_num, node_num)
        query_score_c = torch.nn.functional.normalize(query_score_c, dim = 1, p = 1)
        # query_score_private.append([item.tolist() for item in query_score])
        query_score_common.append(query_score_c.tolist())

        query_score = cosin_similarity(query_feature_private[i], private_embedding_tensor[i])  # (query_num, node_num)
        query_score = torch.nn.functional.normalize(query_score, dim = 1, p = 1)
        # query_score_private.append([item.tolist() for item in query_score])
        query_score_private.append(query_score.tolist())

    # if args.pattern == 'both_std_scores':
    #     for i in range(args.num_view):
    #         query_score_common[i] = torch.from_numpy(standardize_rows(np.array(query_score_common[i])))
    #         query_score_private[i] = torch.from_numpy(standardize_rows(np.array(query_score_private[i])))
    
    for i in range(args.num_view):
        # query_score_common[i] = torch.from_numpy(standardize_rows(np.array(query_score_common[i])))
        # query_score_private[i] = torch.from_numpy(standardize_rows(np.array(query_score_private[i])))
        query_score_common[i] = standardize_rows(np.array(query_score_common[i]))
        query_score_private[i] = standardize_rows(np.array(query_score_private[i]))

    end = time.time()
    print(f'caculate scores using time: {end - start} s')

    return common_embedding_tensor[0].shape[0], query_score_common, query_score_private, graph_list


def main():
    args = parse_args()

    # pattern_list = ['both_std_scores', 'com', 'pri', 'both', 'both_std', ]
    # pattern_list = ['both_std_scores', 'com', 'pri']
    pattern_list = ['both_std_scores']
    result_dict = {
        # 'both': [],
        # 'both_std': [],
        'both_std_scores': []
    }
    nNode, query_score_common, query_score_private, graph_list = load_preprocess_data(args)
    # print('********************************')
    # print('nNode!!!!!!!!!!!!!!!!!!!!!!!')
    # print(nNode)
    # print('**********************************')
    # exit()
    for pt in pattern_list:
        args.pattern = pt
        args.savefile = '_globalsearch_{}_maxIter{}.txt'.format(args.pattern, args.max_iter)
        print('embedding path:', args.EmbeddingPath)
        print('pattern:', args.pattern)
        # lb_range = np.linspace(-2, 2, 41)
        # lb_range = np.array([-1, 0])
        lb_range = np.array([-1])
        # lb_range = [-1, 0]

        if args.pattern == 'com':
            query_score_result = np.array(query_score_common)
        elif args.pattern == 'pri':
            query_score_result = np.array(query_score_private)

        if 'both' in args.pattern:
            # lb_range = np.linspace(0, 1, 11)
            # lb_range = np.linspace(-2, 2, 41)
            for lb in lb_range:
                args.lammbda = lb
                # print(f"query_score_common is {query_score_common}, len(query_score_common) is {len(query_score_common)}, query_score_common[0] is {query_score_common[0]}, query_score_common[0] shape is {query_score_common[0].shape}")
                query_score_result = np.array(query_score_common) + np.array(query_score_private) * args.lammbda
                # query_score_result = torch.stack(query_score_common) + torch.stack(query_score_private) * args.lammbda
                print(args.dataset)
                # 使用 common + private 特征
                res = run(nNode, query_score_result, graph_list, args)
                result_dict[pt].append(res.item())
        else:
            # lb_range = [0]
            args.lammbda = 0
            result_dict[pt] = ['' for _ in range(len(lb_range))]
            result_dict[pt][len(lb_range) // 2] = run(nNode, query_score_result, graph_list, args).item()

        index = lb_range

        # for lb in lb_range:
        #     args.lammbda = lb
        #     res = run(args)
        #     result_dict[pt].append(res)

    df = pd.DataFrame(result_dict, index = index)
    df = df.rename(columns = {'both_std_scores': 'standardize scores'})
    df.to_excel(os.path.join(args.EmbeddingPath, f'aaa_{args.dataset}.xlsx'))



def subgraph_density_controled(candidate_score, graph_score, tau=0.5):
    
    weight_gain = (sum(candidate_score)-sum(graph_score)*(len(candidate_score)**1)/(len(graph_score)**1))/(len(candidate_score)**tau)
    return weight_gain

def GlobalSearch(query_index, graph_score, graph, args):

    candidates = query_index
    selected_candidate = candidates

    graph_score=np.array(graph_score)
    max2min_index = np.argsort(-graph_score)
    
    startpoint = 0
    endpoint = int(0.50*len(max2min_index))
    if endpoint >= 10000:
        endpoint = 10000
    
    while True:
        candidates_half = query_index+[max2min_index[i] for i in range(0, int((startpoint+endpoint)/2))]
        candidate_score_half = [graph_score[i] for i in candidates_half]
        candidates_density_half = subgraph_density_controled(candidate_score_half, graph_score, args.tau)

        candidates = query_index+[max2min_index[i] for i in range(0, endpoint)]
        candidate_score = [graph_score[i] for i in candidates]
        candidates_density = subgraph_density_controled(candidate_score, graph_score, args.tau)

        if candidates_density >= candidates_density_half:
            startpoint = int((startpoint+endpoint)/2)
            endpoint = endpoint
        else:
            startpoint = startpoint
            endpoint = int((startpoint+endpoint)/2)
        
        if startpoint == endpoint or startpoint+1 == endpoint:
            break

    selected_candidate = query_index+[max2min_index[i] for i in range(0, startpoint)] 
    
    return selected_candidate


def run(nNode, query_score_result, graph_list, args, save_result = True):
    query, labels = load_query_n_gt("./dataset/", args.dataset, nNode)
    gt_length = get_gt_legnth("./dataset/", args.dataset)
    # query_score_comPlusPri = np.array(query_score_common) + np.array(query_score_private) * args.lammbda

    use_pretrain = False
    start = time.time()
    if not use_pretrain:
        y_pred_list = []
        

        def process_view(v):
            y_pred = torch.zeros_like(query)
            for i in tqdm(range(query.shape[0])):
                query_index = (torch.nonzero(query[i]).squeeze()).reshape(-1)
                
                selected_candidates = GlobalSearch(
                    query_index.tolist(), query_score_result[v][i], graph_list[v], args
                )
                
                for j in range(len(selected_candidates)):
                    y_pred[i][selected_candidates[j]] = 1
            y_pred_list[v] = y_pred

        threads = []
        y_pred_list = [None] * args.num_view

        for v in range(args.num_view):
            t = threading.Thread(target=process_view, args=(v,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        # np.load(args.EmbeddingPath + args.embedding_tensor_name + f'_pri{i}.npy')
        np.save(os.path.join(
            args.EmbeddingPath, args.embedding_tensor_name + f'_res.npy'), torch.stack(y_pred_list).numpy())

    end = time.time()
    print('Global Search Time:', end - start)
    use_EM = True
    if use_EM:
        print('start EM...')

        max_iter = args.max_iter
        print('max_iter:', max_iter)
        
        y_pred_list = [1 - item for item in y_pred_list]
        y_pred_list = torch.stack(y_pred_list)
        
        # y_pred_list = torch.from_numpy(np.array(y_pred_list))


        srtc = time.time()
        # patients, observers, classes, counts_list = generate_sample_data(y_pred_list)
        counts_list = generate_sample_data(y_pred_list)
        ertc = time.time()
        print('generate data: {} s'.format(ertc - srtc))

        y_pred = torch.zeros_like(query)
        iter_out_number = 0
        start = time.time()
        for i, counts in enumerate(counts_list):
            # resp = 1 - resp
            counts = counts.numpy().astype(np.float64)
            proba_distri, max_iter_out = EMRun(counts, max_iter = max_iter)
            type_distri = np.argmax(proba_distri, axis = 1)
            y_pred[i] = torch.from_numpy(type_distri)
            iter_out_number += int(max_iter_out)
        end = time.time()

        print('EM time: {} s'.format(end - start))
        print('EM iter out num:', iter_out_number)
        labels = 1 - labels
        f1_score = f1_score_calculation(y_pred.int(), labels.int())
        print('common / private: 1 / {}'.format(args.lammbda))
        print("F1 score by maximum weight gain (global search): {:.4f}".format(f1_score))
        if args.write_result:
            with open(os.path.join(
                    args.EmbeddingPath, args.embedding_tensor_name + args.savefile), 'a') as fa:
                fa.write('common / private: 1 / {}, max iter: {}, max iter out: {}\n'.format(args.lammbda, max_iter,
                                                                                             iter_out_number))
                fa.write("F1 score by maximum weight gain (global search): {:.4f}\n\n".format(f1_score))

    return f1_score


def standardize_rows(arr):
    means = np.mean(arr, axis = 1)
    stds = np.std(arr, axis = 1)

    stds[stds == 0] = 1

    standardized_arr = (arr - means[:, np.newaxis]) / stds[:, np.newaxis]

    return standardized_arr


def standardize_columns(arr):
    # 计算每列的均值和标准差
    means = np.mean(arr, axis = 0)
    stds = np.std(arr, axis = 0)

    stds[stds == 0] = 1

    standardized_arr = (arr - means) / stds

    return standardized_arr

def measure_time():
    lb = 1
    args = parse_args()
    print(args.EmbeddingPath)
    args.lammbda = lb
    run(args, save_result = False)


if __name__ == '__main__':
    main()
