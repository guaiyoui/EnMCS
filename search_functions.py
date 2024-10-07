# -*- coding: utf-8 -*-
from functions import find_all_neighbors_bynx


def subgraph_density(candidate_score, avg_weight):
    weight_gain = (sum(candidate_score) - len(candidate_score) * avg_weight) / (len(candidate_score) ** 0.5)
    return weight_gain


def mwg_subgraph_heuristic(query_index, graph_score, graph):
    candidates = query_index

    selected_candidate = candidates
    max_density = -1000

    avg_weight = sum(graph_score) / len(graph_score)

    count = 0
    endpoint = int(0.50 * len(graph_score))
    if endpoint >= 10000:
        endpoint = 10000

    while True:

        neighbors = find_all_neighbors_bynx(candidates, graph)

        if len(neighbors) == 0 or count > endpoint:
            break

        # select the index with the largest score.
        neighbor_score = [graph_score[i] for i in neighbors]
        i_index = neighbor_score.index(max(neighbor_score))

        candidates = candidates + [neighbors[i_index]]

        candidate_score = [graph_score[i] for i in candidates]
        candidates_density = subgraph_density(candidate_score, avg_weight)
        if candidates_density > max_density:
            max_density = candidates_density
            selected_candidate = candidates
        else:
            break

        count += 1

    return selected_candidate


def mwg_subgraph_heuristic_fast_only_common(query_index, graph_score_common, graph_score_private, graph, args):
    candidates = query_index

    selected_candidate = candidates
    max_density = -1000
    max_density_pri = [-1000 for _ in range(args.num_view)]

    avg_weight_com = sum(graph_score_common) / len(graph_score_common)
    avg_weight_pri = [sum(item) / len(item) for item in graph_score_private]

    count = 0
    endpoint = int(0.50 * len(graph_score_common))
    if endpoint >= 10000:
        endpoint = 10000

    threshold = [1 - i / args.num_view for i in range(args.num_view)]
    threshold = [0]
    global_candidates = []
    for t in threshold:
        current_neighbors = find_all_neighbors_bynx(candidates, graph, t)

        current_neighbors_score_com = [graph_score_common[i] for i in current_neighbors]
        candidate_score_com = [graph_score_common[i] for i in candidates]

        current_neighbors_score_pri = []
        candidate_score_pri = []
        for i in range(args.num_view):
            current_neighbors_score_pri.append(
                [graph_score_private[i][j] for j in current_neighbors]
            )
            candidate_score_pri.append(
                [graph_score_private[i][j] for j in candidates]
            )

        while True:

            if len(current_neighbors_score_com) == 0 or count > endpoint:
                break

            i_index = current_neighbors_score_com.index(max(current_neighbors_score_com))


            candidates = candidates + [current_neighbors[i_index]]
            candidate_score_com = candidate_score_com + [graph_score_common[current_neighbors[i_index]]]

            candidates_density = subgraph_density(candidate_score_com, avg_weight_com)

            if candidates_density > max_density:
                # 处理 private，先保证半数以上 view 上的 subgraph_density 增加
                qualified_quantity = 0
                add_flag = False
                save_pri_density = []
                for i in range(args.num_view):
                    temp_cand_pri_score = candidate_score_pri[i] + [graph_score_private[i][current_neighbors[i_index]]]
                    temp_cand_pri_density = subgraph_density(temp_cand_pri_score, avg_weight_pri[i])
                    save_pri_density.append(temp_cand_pri_density)
                    if temp_cand_pri_density > max_density_pri[i]:
                        qualified_quantity += 1
                # if qualified_quantity - args.num_view / 2 >= 1e-9:
                if qualified_quantity - args.num_view / 2 >= 1e-9:
                    add_flag = True
                    del max_density_pri
                    max_density_pri = save_pri_density
                if not add_flag:
                    del current_neighbors[i_index]
                    del current_neighbors_score_com[i_index]
                    continue

                max_density = candidates_density
                selected_candidate = candidates

                new_neighbors = find_all_neighbors_bynx([current_neighbors[i_index]], graph, t)

                del current_neighbors[i_index]
                del current_neighbors_score_com[i_index]

                new_neighbors_unique = list(set(new_neighbors) - set(current_neighbors) - set(candidates))

                new_neighbors_score_com = [graph_score_common[i] for i in new_neighbors_unique]
                current_neighbors = current_neighbors + new_neighbors_unique
                current_neighbors_score_com = current_neighbors_score_com + new_neighbors_score_com

            else:
                break

            count += 1

        global_candidates.extend(selected_candidate)

    return global_candidates
    # return selected_candidate

def greedy_search(query_index, graph_score_common, graph, args):
    candidates = query_index

    selected_candidate = candidates
    max_density = -1000

    avg_weight_com = sum(graph_score_common) / len(graph_score_common)

    count = 0
    endpoint = int(0.50 * len(graph_score_common))
    if endpoint >= 10000:
        endpoint = 10000

    current_neighbors = find_all_neighbors_bynx(candidates, graph, 0)

    current_neighbors_score_com = [graph_score_common[i] for i in current_neighbors]
    candidate_score_com = [graph_score_common[i] for i in candidates]

    while True:
        if len(current_neighbors_score_com) == 0 or count > endpoint:
            break

        i_index = current_neighbors_score_com.index(max(current_neighbors_score_com))
        candidates = candidates + [current_neighbors[i_index]]
        candidate_score_com = candidate_score_com + [graph_score_common[current_neighbors[i_index]]]

        candidates_density = subgraph_density(candidate_score_com, avg_weight_com)

        if candidates_density > max_density:
            max_density = candidates_density
            selected_candidate = candidates

            new_neighbors = find_all_neighbors_bynx([current_neighbors[i_index]], graph, 0)

            del current_neighbors[i_index]
            del current_neighbors_score_com[i_index]

            new_neighbors_unique = list(set(new_neighbors) - set(current_neighbors) - set(candidates))

            new_neighbors_score_com = [graph_score_common[i] for i in new_neighbors_unique]
            current_neighbors = current_neighbors + new_neighbors_unique
            current_neighbors_score_com = current_neighbors_score_com + new_neighbors_score_com

        else:
            break

        count += 1

    return selected_candidate



