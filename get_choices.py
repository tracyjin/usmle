import numpy as np
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import re
import pickle
from pexam_parser import *
from lab_test_parser import *

for mm in range(1, 9):
    test_file = pickle.load(
        open("/Users/xiaomengjin/Desktop/USMLE/USMLE-QA/data/questions/firstaid_qa_step2/t" + str(mm) +
             "q.pkl", "rb"))
    result = []
    for ii in range(len(test_file)):
    	curr_q = test_file[ii]
    	curr_q_choices = curr_q[2]
    	for jj in range(len(curr_q_choices)):
    		result.append(np.array([curr_q_choices[jj], "Hypotension", "contradiction"]))
    np.savetxt('./t' + str(mm) + 'q_choices.tsv', np.array(result), delimiter='\t', fmt='%s')
    	# print(curr_q_choices)



    # print(len(test_file))