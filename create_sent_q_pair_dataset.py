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

def read_data(path):
    f = open(path, "r")
    all_names_dict = {}
    all_ptrs_dict = {}
    all_names_list = []
    for line in f:
        curr_line = line.split("\t")
        all_names_dict[curr_line[1]] = curr_line[2]
        all_names_list.append(curr_line[1])
        if curr_line[2] in all_ptrs_dict:
            all_ptrs_dict[curr_line[2]].append(curr_line[1])
        else:
            all_ptrs_dict[curr_line[2]] = [curr_line[1]]
    return all_names_dict, all_names_list, all_ptrs_dict


all_names_dict, all_names_list, all_ptrs_dict = read_data("./ddb/tblDDB_ItemNames.txt")

def find_exact_match(sent):
    ps = PorterStemmer()
    example = " " + sent.lower() + " "
    curr_example_match = []
    for ii in range(len(all_names_list)):
        curr_word = all_names_list[ii]
        if (" " + curr_word + " ").lower() in example:
            curr_example_match.append(all_names_dict[curr_word])
        elif len(curr_word) > 5 and curr_word.lower() in example:
            curr_example_match.append(all_names_dict[curr_word])
        elif curr_word.lower() in word_tokenize(example):
            curr_example_match.append(all_names_dict[curr_word])
        elif ps.stem(curr_word).lower() in word_tokenize(example):
            curr_example_match.append(all_names_dict[curr_word])
    return curr_example_match


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
    node_data = read_data_for_graph("./ddb/tblDDB_ItemNames.txt")
    edge_data = read_data_for_graph("./ddb/tblDDB_ItemRelationships.txt")
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


def get_shortest_path(node1, node2):
    ret_path = ''
    len_path = None
    all_paths = nx.all_simple_paths(G, node1, node2, cutoff=3)
    to_list = list(all_paths)
    # count = 0
    # results = [[], []]
    # for path in map(nx.utils.pairwise, all_paths):
    #     # print(results[0], to_list[count])
    #     print(to_list[count])
    #     print(path)
    #     results[0].append([count])
    #     relation_path = []
    #     for edge in list(path):
    #         relation_path.append(int(G.edges[edge]['info'][0]['ItemRelationshipTypePTR']))
    #     results[1].append(list(relation_path))
    #     count += 1
    results = [[], []]
    for ii in range(len(to_list)):
        path = to_list[ii]
        results[0].append(path)
        relation_path = []
        for kk in range(len(path) - 1):
            relation_path.append(int(G.edges[(path[kk], path[kk + 1])]['info'][0]['ItemRelationshipTypePTR']))
        results[1].append(list(relation_path))
    return results


all_question_matches = []
G = build_graph()
answers = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8}

for mm in range(1, 9):
    test_file = pickle.load(
        open("/Users/xiaomengjin/Desktop/USMLE/USMLE-QA/data/questions/firstaid_qa_step2/t" + str(mm) +
             "q.pkl", "rb"))
    answer_file = pickle.load(
        open("/Users/xiaomengjin/Desktop/USMLE/USMLE-QA/data/questions/firstaid_qa_step2/t" + str(mm) +
             "a.pkl", "rb"))
    after_tests_parser = []

    ddb_names = np.load("./predictions_outputs_t" + str(mm) + "q_need_match/test_names.npz")["names"]
    f = open("t" + str(mm) + "q_need_match.tsv", "r")
    all_bert_sent_dicts = {}
    count = 0
    for line in f:
        curr_line = line.split("\t")
        curr_line[-1] = curr_line[-1][:-1]
        curr_logits = np.load(
            "./predictions_outputs_t" + str(mm) + "q_need_match/test_example_" +
            str(count) + ".npz")["logits"]
        sorted_logits_ind = np.argsort(curr_logits)[::-1]
        top_3_ptr = []
        for jj in range(100):
            curr_name = ddb_names[sorted_logits_ind[jj]]
            curr_ptr = all_names_dict[curr_name]
            if curr_ptr not in top_3_ptr:
                top_3_ptr.append(curr_ptr)
            if len(top_3_ptr) == 3:
                break
        all_bert_sent_dicts[curr_line[0]] = top_3_ptr
        count += 1
    f = open("t" + str(mm) + "q_choices.tsv", "r")
    all_bert_choice_dicts = {}
    count = 0
    for line in f:
        curr_line = line.split("\t")
        curr_line[-1] = curr_line[-1][:-1]
        curr_logits = np.load(
            "./predictions_outputs_t" + str(mm) + "q_choices/test_example_" +
            str(count) + ".npz")["logits"]
        sorted_logits_ind = np.argsort(curr_logits)[::-1]
        top_3_ptr = []
        for jj in range(100):
            curr_name = ddb_names[sorted_logits_ind[jj]]
            curr_ptr = all_names_dict[curr_name]
            if curr_ptr not in top_3_ptr:
                top_3_ptr.append(curr_ptr)
            if len(top_3_ptr) == 3:
                break
        all_bert_choice_dicts[curr_line[0]] = top_3_ptr
        count += 1
    for ii in range(len(test_file)):
        matched_ddb_des = []
        matched_ddb_choices = []
        need_match = []
        curr_q = test_file[ii]
        correct_choice = answers[answer_file[ii][0]]
        # sentence list storing the problem description
        curr_q_des = sent_tokenize(curr_q[0])[1:-1]
        # string question sentence
        curr_q_q = sent_tokenize(curr_q[0])[-1]
        # test value list extracted from the table
        curr_q_lst = curr_q[1]
        # all choices list
        curr_q_choices = curr_q[2]
        for kk in range(len(curr_q_des)):
            # two lists, has the mapping, the second inidates negation (value: 1)
            vitals = match_vitals(curr_q_des[kk].replace(",", ""))
            lab_tests = match_lab_tests(curr_q_des[kk].replace(",", ""))
            if len(lab_tests[0]) != 0:
                for tt in range(len(lab_tests[0])):
                    matched_ddb_des.append(all_names_dict[lab_tests[0][tt]])
                    # matched_ddb_des.append(all_names_dict[lab_tests[0][tt]])
            if len(vitals[0]) != 0:
                for tt in range(len(vitals[0])):
                    matched_ddb_des.append(all_names_dict[vitals[0][tt]])
                    # matched_ddb_des.append(all_names_dict[vitals[0][tt]])
            if len(vitals[0]) == 0 and len(lab_tests[0]) == 0:
                need_match.append(curr_q_des[kk])
        for kk in range(len(curr_q_lst)):
            lab_tests = match_lab_tests(curr_q_lst[kk].replace(",", ""))
            if len(lab_tests[0]) != 0:
                for tt in range(len(lab_tests[0])):
                    if lab_tests[0][tt] is not None:
                        matched_ddb_des.append(all_names_dict[lab_tests[0][tt]])
                        # matched_ddb_des.append(all_names_dict[lab_tests[0][tt]])
        for kk in range(len(need_match)):
            curr_sent = need_match[kk]
            exact_match = find_exact_match(curr_sent)
            if len(exact_match) != 0:
                matched_ddb_des += exact_match
            else:
                if curr_sent in all_bert_sent_dicts:
                    matched_ddb_des += all_bert_sent_dicts[curr_sent]
        matched_ddb_des = list(set(matched_ddb_des))
        for kk in range(len(curr_q_choices)):
            curr_choice = curr_q_choices[kk]
            exact_match = find_exact_match(curr_choice)
            if len(exact_match) != 0:
                matched_ddb_choices.append(exact_match)
            else:
                if curr_choice in all_bert_choice_dicts:
                    matched_ddb_choices.append(all_bert_choice_dicts[curr_choice])
                else:
                    matched_ddb_choices.append([])
        question_results = [correct_choice]
        for jj in range(len(matched_ddb_choices)):
            curr_choice_result = [[], []]
            for nn in range(len(matched_ddb_choices[jj])):
                for kk in range(len(matched_ddb_des)):
                    curr_result = get_shortest_path(matched_ddb_choices[jj][nn], matched_ddb_des[kk])
                    curr_choice_result[0] += curr_result[0]
                    curr_choice_result[1] += curr_result[1]
            question_results.append(curr_choice_result)
            # print(question_results)
            # raise
        all_question_matches.append(question_results)

        print(len(all_question_matches))



    with open('./data.pkl', 'wb') as fp:
        pickle.dump(all_question_matches, fp)


raise






