import numpy as np
import pickle




data_file = pickle.load(open("./data.pkl", "rb"))

relation_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 10: 6, 12: 7, 16: 8, 17: 9, 18: 10, 20: 11, 26: 12, 30: 13}

all_data = []

all_labels = []


count = 0
for question in data_file:
    label = question[0]
    count += 1
    curr_q = []
    for ii in range(1, len(question)):
        curr_new_data = np.zeros(14 + 14 ** 2 + 14 ** 3)
        all_paths = question[ii][1]
        # print(label)
        for path in all_paths:
            if len(path) == 1:
                curr_new_data[relation_dict[path[0]]] = 1
            elif len(path) == 2:
                ind = relation_dict[path[0]] * 14 + relation_dict[path[1]] + 14
                curr_new_data[ind] = 1
            elif len(path) == 3:
                ind = relation_dict[path[0]] * 14 ** 2 + relation_dict[path[1]] * 14 + relation_dict[path[2]] + 14 ** 2 + 14
                curr_new_data[ind] = 1
        curr_q.append(np.array(curr_new_data))
    all_data.append(np.array(curr_q))
    all_labels.append(label)
# raise
# count = 0
# for mm in range(1, 9):
#     test_file = pickle.load(
#         open("/Users/xiaomengjin/Desktop/USMLE/USMLE-QA/data/questions/firstaid_qa_step2/t" + str(mm) +
#              "q.pkl", "rb"))
#     for q in test_file:
#         count += 1
#         if count >= 28 and count <= 32:
#             print(count, q)
all_data = np.array(all_data)
np.savez("./path_relation_type.npz", x=all_data, y=all_labels)

print(all_data.shape)
# print(np.sum(all_data))




