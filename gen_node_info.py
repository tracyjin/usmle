import numpy as np
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import re
import pickle
from pexam_parser import *
from lab_test_parser import *
import networkx as nx

import random

random.seed(66666)
np.random.seed(66666)

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
    G = nx.DiGraph()
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
        if edge[-1] == '0' or edge[-1] == '1':
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
    return G, all_nodes

G, all_nodes = build_graph()


def get_shortest_path(node1):
    results = nx.single_source_shortest_path(G, node1)
    return results

not_found_types = []

exists = {'33486': 0, '28809': 1, '28811': 2, '28855': 3, '33130': 4, '28830': 5, '28828': 6, '28831': 7, '28832': 8, '28836': 9, '28857': 10, '28848': 11, '30693': 12, '28856': 13, '36594': 1, '33461': 14, '36318': 14, '34397': 13, '34528': 14, '35157': 13}


all_node_types = {}

for node in G.nodes:
    # print(node)
    # raise
    # print(node[2])
    result = get_shortest_path(node)
    found = False
    for name in result.keys():
        if name in exists:
            all_node_types[node] = exists[name]
            found = True
            break
    if found == False:
        not_found_types.append(node)



print(len(all_node_types.keys()))
print(len(G.nodes))
not_found_types = list(set(not_found_types))

# for node in not_found_types:
#     all_node_types[node] = all_node_types[name]



print(set(not_found_types))


NODE_TYPE_MAPS = all_node_types
# all_types = list(set(all_types))

# for curr_type in all_types:
#     for info in G.nodes[curr_type]['info']:
#         if info['IsPreferredName'] == '1':
#             print(info['ItemName'])

# print(len(list(set(all_types))))







