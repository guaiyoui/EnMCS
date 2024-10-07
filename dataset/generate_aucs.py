# import numpy as np
# import random
# def generate_query_nodes(file_path, output_file, output_file_gt, all_communities=True):
#     with open(file_path, 'r') as f:
#         communities = [line.strip().split(' ') for line in f.readlines()]

#     num_communities = len(communities)
#     selected_communities = range(num_communities) if all_communities else random.sample(range(num_communities), num_communities // 2)

#     outpyt_file_gt_file = open(output_file_gt, 'w')

#     with open(output_file, 'w') as f:
#         count = 0
#         while count < 100:
#             for community_idx in selected_communities:
#                 community = communities[community_idx]
#                 num_nodes = random.randint(1, 3)
#                 if num_nodes > len(community):
#                     num_nodes = len(community)
#                 selected_nodes = random.sample(community, num_nodes)
#                 for node in selected_nodes:
#                     f.write(f"{int(node)-1} ")
#                 # f.write(f"{community_idx}\n")
#                 f.write(f"\n")

#                 for item in community:
#                     outpyt_file_gt_file.write(f"{int(item)-1} ")
#                 outpyt_file_gt_file.write(f"\n")
#                 # f.write(f"{community_idx}\n")
#                 # # for node in selected_nodes:
#                 # f.write(f"{selected_nodes} {community_idx}\n")
#                 count += 1
#                 if count >= 100:
#                     break

# directory = 'aucs_communities.txt'
# output_dir = 'aucs_query_nodes'
# generate_query_nodes(directory, './aucs/aucs.query', './aucs/aucs.gt', all_communities=True, )


import numpy as np
import random
def generate_query_nodes(file_path, output_file, output_file_gt, all_communities=True):
    with open(file_path, 'r') as f:
        communities = [line.strip().split(' ') for line in f.readlines()]

    num_communities = len(communities)
    selected_communities = range(num_communities) if all_communities else random.sample(range(num_communities), num_communities // 2)

    outpyt_file_gt_file = open(output_file_gt, 'w')

    with open(output_file, 'w') as f:
        count = 0
        while count < 5000:
            for community_idx in selected_communities:
                community_idx = 1
                community = communities[community_idx]
                num_nodes = random.randint(1, 5)
                if num_nodes > len(community):
                    num_nodes = len(community)
                selected_nodes = random.sample(community, num_nodes)
                for node in selected_nodes:
                    f.write(f"{int(node)-1} ")
                # f.write(f"{community_idx}\n")
                f.write(f"\n")

                for item in community:
                    outpyt_file_gt_file.write(f"{int(item)-1} ")
                outpyt_file_gt_file.write(f"\n")
                # f.write(f"{community_idx}\n")
                # # for node in selected_nodes:
                # f.write(f"{selected_nodes} {community_idx}\n")
                count += 1
                if count >= 100:
                    break

directory = 'aucs_communities.txt'
output_dir = 'aucs_query_nodes'
generate_query_nodes(directory, './aucs/aucs.query', './aucs/aucs.gt', all_communities=True, )
