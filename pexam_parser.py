#!/usr/bin/python

import re
import math

normal_vitals = {re.compile("(?<=[Tt]emperature is )[34][0-9.]*"): 
                    [[36.1, 37.2], ["Hypothermia", "Fever"], "temperature"],
                 re.compile("(?<=[Tt]emperature )[34][0-9.]*"):
                    [[36.1, 37.2], ["Hypothermia", "Fever"], "temperature"],
                 re.compile("(?<=[Ff]ever of )[34][0-9.]*"):
                    [[36.1, 37.2], ["Hypothermia", "Fever"], "temperature"],
                 re.compile("(?<=\()[891][0-9.](?=F\))"): #need some polish
                        [[97, 99], ["Hypothermia", "Fever"], "temperature_F"],
                 re.compile("(?<=[Pp]ulse is )[0-9]+"): 
                        [[60, 100], ["Sinus bradycardia", "Sinus tachycardia"], "pulse"],
                 re.compile("(?<=[Pp]ulse )[0-9]+"): 
                        [[60, 100], ["Sinus bradycardia", "Sinus tachycardia"], "pulse"],
                 re.compile("(?<=[Hh]eart rate is )[0-9]+"): 
                        [[60, 100], ["Sinus bradycardia", "Sinus tachycardia"], "pulse"],
                 #note: multiple matchings for tachycardia
                 re.compile("(?<=[Rr]espirations )[0-9]+"): 
                        [[12, 20], ["Oligopnea", "Dyspnoea"], "respirations"],
                 re.compile("(?<=[Rr]espirations are )[0-9]+"): 
                        [[12, 20], ["Oligopnea", "Dyspnoea"], "respirations"],
                 re.compile("(?<=[Rr]espiratory rate is )[0-9]+"): 
                        [[12, 20], ["Oligopnea", "Dyspnoea"], "respirations"],
                 #TODO: Bradypnea not in DDB
                 re.compile("(?<=[Bb]lood pressure )[0-9]+(?= /)"):
                        [[100, 140], ["Hypotension", "Hypertension, systemic"], "SBP"],
                 re.compile("(?<=[Bb]lood pressure is )[0-9]+(?= /)"):
                        [[100, 140], ["Hypotension", "Hypertension, systemic"], "SBP"],
                 re.compile("(?<=/)[0-9]+(?= mm Hg)"):  #nned some polish
                        [[60, 90], ["Hypotension", "Hypertension, systemic"], "DBP"],
                 re.compile("(?<=BMI [i][s][ ])[0-9]+(?= kg/m)"): 
                        [[18.5, 25],["Body mass index low", 
                        "Body mass index raised"], "BMI"],

                 re.compile("(?<=is )[0-9]+(?= cm)"):
                    [[0, math.inf],"", "height"], # we ignore h/w here 
                 re.compile("(?<=weighs )[0-9]+(?= kg)"): 
                    [[0, math.inf], "", "weight"],
                 #>100 is meaningless but let us avoid pitfall here
                 re.compile("(0-9)+(?= %)"):
                 [[95, math.inf], ["Hypoxemia"], "oxygen saturation"]}
            

#normal values
pe_all_negations_list = ["Physical examination shows no abnormalities", 
                    "Examination shows no abnormalities"]

pe_negations = {"lungs are clear to auscultation": 
                ["Breath sounds absent", 
                 "Breath sounds bronchial", 
                 "Breath sounds reduced"],
                 "normal heart sounds": ["Abnormal heart sounds", 
                 "Abnormal splitting of heart sounds", 
                 "Fourth heart sound", "Third heart sound",
                 "Fixed split of second heart sound",
                 "Paradoxic split of second heart sound",
                 "Wide split of second heart sound"],
                 "abdomen is soft": ["abdominal guarding"], #TODO: not in DDB
                 "soft abdoment":  ["abdominal guarding"]}

pe_all_negations = {}
for phrase in pe_all_negations_list:
    pe_all_negations[phrase] = pe_negations.values()

general_negations = {"normal", "no abnormalities", 
                     "within the reference ranges"}

#families
first_degree_families = ["mother", "father", "parents", 
                         "daughter", "son", "kid", "child",
                         "brother", "sister", "sibling"]
second_degree_families = ["grandmother", "grandfather", 
                          "grandparent", "aunt", "uncle",
                          "niece", "nephew"]
third_degree_families = ["cousin"]

history = ["history", "years ago", "months ago", "days ago", "since age"]


#returns a list of mapped negations
#notice they are all negations!
def match_pe(sentence):
    return_val = []
    for k, v in pe_negations.items():
        if k in sentence:
            return_val += v
    return(return_val)
    

#returns two lists, has the mapping, the second inidates negation (value: 1)
def match_vitals(sentence):
    return_lists = [[], []]
    for k, v in normal_vitals.items():
        m = k.findall(sentence)
        if m:
            if v[2] == "height" or v[2] == "weight": continue
            if float(m[0]) < v[0][0]: #low
                return_lists[0].append(v[1][0])
                return_lists[1].append(0)
            elif float(m[0]) > v[0][1]: #high
                return_lists[0].append(v[1][1])
                return_lists[1].append(0)
            else: #normal
                return_lists[0] += v[1]
                return_lists[1] += [1, 1] #two negations
    return(return_lists)           
            
    
# returns a list with first value indicating history 
# and the second value indicating negation 
# and the third value indicating familes (with degrees)
def match_hx_neg(sentence):
    return_value = [0, 0, 0]

    for phrase in history:
        if phrase in sentence:
            return_value[0] = 1
    for phrase in general_negations:
        if phrase in sentence:
            return_value[1] = 1

    #TODO: we do not map multiple members yet
    for phrase in first_degree_families:
        if phrase in sentence:
            return_value[2] = 1
    for phrase in second_degree_families:
        if phrase in sentence:
            return_value[2] = 2
    for phrase in third_degree_families:
        if phrase in sentence:
            return_value[2] = 3
            
    return(return_value)
    
