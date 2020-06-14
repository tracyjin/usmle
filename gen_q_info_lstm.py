import numpy as np
import pickle
import networkx as nx
from gen_node_info import NODE_TYPE_MAPS

def read_data_for_graph(path):
    f = open(path, "r")
    all_lines = []
    for line in f:
        curr_line = line.split("\t")
        curr_line[-1] = curr_line[-1][:-1]
        all_lines.append(curr_line)
    return all_lines

def build_attr(data, attr_name):
    ret_dict = {}
    for ii in range(len(data)):
        ret_dict[attr_name[ii]] = data[ii]
    return ret_dict


def build_graph():
    G = nx.Graph()
    node_data = read_data_for_graph("./dataset/ddb/tblDDB_ItemNames.txt")
    edge_data = read_data_for_graph("./dataset/ddb/tblDDB_ItemRelationships.txt")
    node_attr, all_nodes = node_data[0], node_data[1:]
    edge_attr, all_edges = edge_data[0], edge_data[1:]
    all_symp_names = []
    all_symp_ids = []
    for node in all_nodes:
        curr_node_attr = build_attr(node, node_attr)
        all_symp_names.append(node[1].lower())
        all_symp_ids.append(node[2])
        if G.has_node(node[2]):
            G.nodes[node[2]]['info'] += [curr_node_attr]
        else:
            G.add_node(node[2], info=[curr_node_attr])
    for edge in all_edges:
        if edge[-1] != '0' and edge[-1] != '1':
            curr_edge_attr = build_attr(edge, edge_attr)
            if G.has_edge(edge[1], edge[2]):
                G.edges[(edge[1], edge[2])]['info'] += [curr_edge_attr]
            else:
                G.add_edge(edge[1], edge[2], info=[curr_edge_attr])
    for edge in G.edges:
        s = edge[0]
        e = edge[1]
        # edge_w = int(np.sqrt(G.degree[s])) * int(np.sqrt(G.degree[e]))
        # edge_w = np.sqrt(np.sqrt(1.0 * G.degree[s]) * np.sqrt(1.0 * G.degree[e]))
        edge_w = np.sqrt(1.0 * G.degree[s]) * np.sqrt(1.0 * G.degree[e])
        G.edges[edge]['weight'] = edge_w
    return G


G = build_graph()


data_file = pickle.load(open("./data.pkl", "rb"))

all_q_result_train = []
all_labels_train = []
all_q_path_len_train = []

all_q_result_test = []
all_labels_test = []
all_q_path_len_test = []


# for question in data_file:
#     label = question[0]
#     curr_q_new_pairs = []
#     curr_q_path_len = []
#     for ii in range(1, len(question)):
#         all_paths = question[ii][0]
#         all_edges = question[ii][1]
#         curr_choice_new_path_pairs = []
#         curr_choice_path_len = []
#         # print(len(all_paths))
#         # raise
#         for kk in range(len(all_paths)):
#             curr_path = all_paths[kk]
#             curr_edges = all_edges[kk]
#             curr_new_pair = []
#             for mm in range(len(curr_path) - 1):
#                 curr_start_node = curr_path[mm]
#                 curr_end_node = curr_path[mm + 1]
#                 curr_edge = curr_edges[mm]
#                 curr_new_pair.append(np.array([NODE_TYPE_MAPS[curr_start_node], G.degree[curr_start_node], curr_edge, NODE_TYPE_MAPS[curr_end_node], G.degree[curr_end_node]]))
#             for mm in range(len(curr_path) - 1, 3):
#                 curr_new_pair.append(np.array([0, 0, 0, 0, 0]))
#             # print(len(curr_path))
#             curr_choice_new_path_pairs.append(np.array(curr_new_pair))
#             curr_choice_path_len.append(len(curr_path) - 1)
#         curr_q_new_pairs.append(np.array(curr_choice_new_path_pairs))
#         curr_q_path_len.append(np.array(curr_choice_path_len))
#     all_q_result.append(np.array(curr_q_new_pairs))
#     all_q_path_len.append(np.array(curr_q_path_len))
#     all_labels.append(label)

# np.savez("./q_info_lstm.npz", x=np.array(all_q_result), y=np.array(all_labels), path_len=np.array(all_q_path_len))
# print(all_q_result)
relation_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 10: 6, 12: 7, 16: 8, 17: 9, 18: 10, 20: 11, 26: 12, 30: 13}


rand_lst = np.arange(len(data_file))
np.random.shuffle(rand_lst)
# rand_lst = list(rand_lst)
print(rand_lst)

count = 0

for question in data_file:
    label = question[0]
    curr_q_new_pairs = []
    curr_q_path_len = []
    # if len(question) <= 2:
    #     raise
    # if count == 79:
    #     print(question)
    for ii in range(1, len(question)):
        all_paths = question[ii][0]
        all_edges = question[ii][1]
        curr_choice_new_path_pairs = []
        curr_choice_path_len = []
        # print(len(all_paths))
        # raise
        for kk in range(len(all_paths)):
            curr_path = all_paths[kk]
            curr_edges = all_edges[kk]
            curr_new_pair = []
            for mm in range(len(curr_path) - 1):
                curr_start_node = curr_path[mm]
                curr_end_node = curr_path[mm + 1]
                curr_edge = curr_edges[mm]
                feat_list = [0] * 44
                feat_list[NODE_TYPE_MAPS[curr_start_node]] = 1
                feat_list[15] = G.degree[curr_start_node]
                # print(curr_edge)
                feat_list[16 + relation_dict[curr_edge]] = 1
                feat_list[28 + NODE_TYPE_MAPS[curr_end_node]] = 1
                feat_list[-1] = G.degree[curr_end_node]
                # curr_new_pair.append(np.array([NODE_TYPE_MAPS[curr_start_node], G.degree[curr_start_node], curr_edge, NODE_TYPE_MAPS[curr_end_node], G.degree[curr_end_node]]))
                curr_new_pair.append(np.array(feat_list))
            for mm in range(len(curr_path) - 1, 3):
                # curr_new_pair.append(np.array([0, 0, 0, 0, 0]))
                feat_list = [0] * 44
                curr_new_pair.append(np.array(feat_list))
            # print(len(curr_path))
            curr_choice_new_path_pairs.append(np.array(curr_new_pair))
            # curr_q_new_pairs.append(np.array(curr_new_pair))
            curr_choice_path_len.append(len(curr_path) - 1)
        curr_q_new_pairs += curr_choice_new_path_pairs
        curr_q_path_len.append(np.array(curr_choice_path_len))
    if count in rand_lst[:int(0.8 * len(data_file))]:
        all_q_result_train.append(np.array(curr_q_new_pairs))
        all_q_path_len_train.append(np.array(curr_q_path_len))
        all_labels_train.append(label)
    else:
        all_q_result_test.append(np.array(curr_q_new_pairs))
        all_q_path_len_test.append(np.array(curr_q_path_len))
        all_labels_test.append(label)
    count += 1


np.savez("./q_info_lstm_all.npz",
    x_train=np.array(all_q_result_train), y_train=np.array(all_labels_train), path_len_train=np.array(all_q_path_len_train),
    x_test=np.array(all_q_result_test), y_test=np.array(all_labels_test), path_len_test=np.array(all_q_path_len_test))
print(all_q_result_train[78:80])





