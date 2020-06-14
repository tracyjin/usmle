#!/usr/bin/python

import re
import math

lab_test = {re.compile("(?<=[Aa]lanine aminotransferase is )[0-9.]* (?=U/L)"): 
            [[8, 20], [None, "ALT raised"], "ALT"],
            re.compile("(?<=ALT is )[0-9.]* (?=U/L)"): 
            [[8, 20], [None, "ALT raised"], "ALT"],
            re.compile("(?<=[Aa]lanine aminotransferase: )[0-9.]* (?=U/L)"): 
            [[8, 20], [None, "ALT raised"], "ALT"],
            re.compile("(?<=ALT: )[0-9.]* (?=U/L)"): 
            [[8, 20], [None, "ALT raised"], "ALT"],
            re.compile("(?<=[Aa]mylase is )[0-9.]* (?=U/L)"): 
            [[25, 125], ["Amylase levels low (plasma or serum)", "Amylase levels raised (plasma or serum)"], "Amylase"],
            re.compile("(?<=[Aa]mylase: )[0-9.]* (?=U/L)"): 
            [[25, 125], ["Amylase levels low (plasma or serum)", "Amylase levels raised (plasma or serum)"], "Amylase"],
            re.compile("(?<=[Aa]spartate aminotransferase is )[0-9.]* (?=U/L)"): 
            [[8, 20], [None, "AST raised"], "AST"],
            re.compile("(?<=AST is )[0-9.]* (?=U/L)"): 
            [[8, 20], [None, "AST raised"], "AST"],
            re.compile("(?<=[Aa]spartate aminotransferase: )[0-9.]* (?=U/L)"): 
            [[8, 20], [None, "AST raised"], "AST"],
            re.compile("(?<=AST: )[0-9.]* (?=U/L)"): 
            [[8, 20], [None, "AST raised"], "AST"],
            re.compile("(?<=Bilirubin \([Tt]otal\): )[0-9.]* (?=mg/dL)"): 
            [[0.1, 1.0], [None, "Bilirubin levels raised (plasma or serum)"], "Bilirubin Total"],
            re.compile("(?<=[Tt]otal Bilirubin: )[0-9.]* (?=mg/dL)"): 
            [[0.1, 1.0], [None, "Bilirubin levels raised (plasma or serum"], "Bilirubin Total"],
            re.compile("(?<=Bilirubin \([Tt]otal\): )[0-9.]* (?=μmol/L)"): 
            [[2, 17], [None, "Bilirubin levels raised (plasma or serum)"], "Bilirubin Total"],
            re.compile("(?<=[Tt]otal Bilirubin: )[0-9.]* (?=μmol/L)"): 
            [[2, 17], [None, "Bilirubin levels raised (plasma or serum"], "Bilirubin Total"],
            re.compile("(?<=Bilirubin \([Dd]irect\): )[0-9.]* (?=mg/dL)"): 
            [[0.0, 0.3], [None, "Bilirubin levels raised (plasma or serum)"], "Bilirubin Direct"],
            re.compile("(?<=[Dd]irect Bilirubin: )[0-9.]* (?=mg/dL)"): 
            [[0.0, 0.3], [None, "Bilirubin levels raised (plasma or serum"], "Bilirubin Direct"],
            re.compile("(?<=Bilirubin \([Dd]irect\): )[0-9.]* (?=μmol/L)"): 
            [[0, 5], [None, "Bilirubin levels raised (plasma or serum)"], "Bilirubin Direct"],
            re.compile("(?<=[Dd]irect Bilirubin: )[0-9.]* (?=μmol/L)"): 
            [[0, 5], [None, "Bilirubin levels raised (plasma or serum"], "Bilirubin Direct"],
            
            re.compile("(?<=[Cc]alcium [Ss]erum is )[0-9.]* (?=mg/dL)"): 
            [[8.4, 10.2], ["Calcium levels low (plasma)", "Calcium levels raised (plasma)"], "Calcium Serum"],
            re.compile("(?<=[Cc]alcium [Ss]erum: )[0-9.]* (?=mg/dL)"): 
            [[8.4, 10.2], ["Calcium levels low (plasma)", "Calcium levels raised (plasma)"], "Calcium Serum"],
            re.compile("(?<=[Ss]erum [Cc]alcium is )[0-9.]* (?=mg/dL)"): 
            [[8.4, 10.2], ["Calcium levels low (plasma)", "Calcium levels raised (plasma)"], "Calcium Serum"],
            re.compile("(?<=[Ss]erum [Cc]alcium: )[0-9.]* (?=mg/dL)"): 
            [[8.4, 10.2], ["Calcium levels low (plasma)", "Calcium levels raised (plasma)"], "Calcium Serum"],
            re.compile("(?<=[Cc]alcium [Ss]erum is )[0-9.]* (?=mmol/L)"): 
            [[2.1, 2.8], ["Calcium levels low (plasma)", "Calcium levels raised (plasma)"], "Calcium Serum"],
            re.compile("(?<=[Cc]alcium [Ss]erum: )[0-9.]* (?=mmol/L)"): 
            [[2.1, 2.8], ["Calcium levels low (plasma)", "Calcium levels raised (plasma)"], "Calcium Serum"],
            re.compile("(?<=[Ss]erum [Cc]alcium is )[0-9.]* (?=mmol/L)"): 
            [[2.1, 2.8], ["Calcium levels low (plasma)", "Calcium levels raised (plasma)"], "Calcium Serum"],
            re.compile("(?<=[Ss]erum [Cc]alcium: )[0-9.]* (?=mmol/L)"): 
            [[2.1, 2.8], ["Calcium levels low (plasma)", "Calcium levels raised (plasma)"], "Calcium Serum"],
            
            re.compile("(?<=[Ss]erum [Cc]alcium [Cc]oncentration is )[0-9.]* (?=mg/dL)"): 
            [[8.4, 10.2], ["Calcium levels low (plasma)", "Calcium levels raised (plasma)"], "Calcium Serum"],
            re.compile("(?<=[Ss]erum [Cc]alcium [Cc]oncentration is )[0-9.]* (?=mmol/L)"): 
            [[2.1, 2.8], ["Calcium levels low (plasma)", "Calcium levels raised (plasma)"], "Calcium Serum"],
            


            re.compile("(?<=[Cc]a2\+ [Ss]erum is )[0-9.]* (?=mg/dL)"): 
            [[8.4, 10.2], ["Calcium levels low (plasma)", "Calcium levels raised (plasma)"], "Calcium Serum"],
            re.compile("(?<=[Cc]a2\+ [Ss]erum: )[0-9.]* (?=mg/dL)"): 
            [[8.4, 10.2], ["Calcium levels low (plasma)", "Calcium levels raised (plasma)"], "Calcium Serum"],
            re.compile("(?<=[Ss]erum [Cc]a2\+ is )[0-9.]* (?=mg/dL)"): 
            [[8.4, 10.2], ["Calcium levels low (plasma)", "Calcium levels raised (plasma)"], "Calcium Serum"],
            re.compile("(?<=[Ss]erum [Cc]a2\+: )[0-9.]* (?=mg/dL)"): 
            [[8.4, 10.2], ["Calcium levels low (plasma)", "Calcium levels raised (plasma)"], "Calcium Serum"],
            re.compile("(?<=[Cc]a2\+ [Ss]erum is )[0-9.]* (?=mmol/L)"): 
            [[2.1, 2.8], ["Calcium levels low (plasma)", "Calcium levels raised (plasma)"], "Calcium Serum"],
            re.compile("(?<=[Cc]a2\+ [Ss]erum: )[0-9.]* (?=mmol/L)"): 
            [[2.1, 2.8], ["Calcium levels low (plasma)", "Calcium levels raised (plasma)"], "Calcium Serum"],
            re.compile("(?<=[Ss]erum [Cc]a2\+ is )[0-9.]* (?=mmol/L)"): 
            [[2.1, 2.8], ["Calcium levels low (plasma)", "Calcium levels raised (plasma)"], "Calcium Serum"],
            re.compile("(?<=[Ss]erum [Cc]a2\+: )[0-9.]* (?=mmol/L)"): 
            [[2.1, 2.8], ["Calcium levels low (plasma)", "Calcium levels raised (plasma)"], "Calcium Serum"],


            re.compile("(?<=[Cc]holesterol [Ss]erum is )[0-9.]* (?=mg/dL)"): 
            [[0, 200], ["Cholesterol levels low (plasma or serum)", "Cholesterol levels raised (plasma or serum)"], "Cholesterol Serum"],
            re.compile("(?<=[Cc]holesterol [Ss]erum: )[0-9.]* (?=mg/dL)"): 
            [[0, 200], ["Cholesterol levels low (plasma or serum)", "Cholesterol levels raised (plasma or serum)"], "Cholesterol Serum"],
            re.compile("(?<=[Ss]erum [Cc]holesterol is )[0-9.]* (?=mg/dL)"): 
            [[0, 200], ["Cholesterol levels low (plasma or serum)", "Cholesterol levels raised (plasma or serum)"], "Cholesterol Serum"],
            re.compile("(?<=[Ss]erum [Cc]holesterol: )[0-9.]* (?=mg/dL)"): 
            [[0, 200], ["Cholesterol levels low (plasma or serum)", "Cholesterol levels raised (plasma or serum)"], "Cholesterol Serum"],
            re.compile("(?<=[Cc]holesterol [Ss]erum is )[0-9.]* (?=mmol/L)"): 
            [[0, 5.2], ["Cholesterol levels low (plasma or serum)", "Cholesterol levels raised (plasma or serum)"], "Cholesterol Serum"],
            re.compile("(?<=[Cc]holesterol [Ss]erum: )[0-9.]* (?=mmol/L)"): 
            [[0, 5.2], ["Cholesterol levels low (plasma or serum)", "Cholesterol levels raised (plasma or serum)"], "Cholesterol Serum"],
            re.compile("(?<=[Ss]erum [Cc]holesterol is )[0-9.]* (?=mmol/L)"): 
            [[0, 5.2], ["Cholesterol levels low (plasma or serum)", "Cholesterol levels raised (plasma or serum)"], "Cholesterol Serum"],
            re.compile("(?<=[Ss]erum [Cc]holesterol: )[0-9.]* (?=mmol/L)"): 
            [[0, 5.2], ["Cholesterol levels low (plasma or serum)", "Cholesterol levels raised (plasma or serum)"], "Cholesterol Serum"],
            
            re.compile("(?<=[Cc]ortisol, [Ss]erum 0800h is )[0-9.]* (?=μg/dL)"): 
            [[5, 23], ["Cortisol levels low (serum or plasma)", "Cortisol levels raised (serum or plasma)"], "Cortisol Serum"],
            re.compile("(?<=[Cc]ortisol, [Ss]erum 0800h: )[0-9.]* (?=μg/dL)"): 
            [[5, 23], ["Cortisol levels low (serum or plasma)", "Cortisol levels raised (serum or plasma)"], "Cortisol Serum"],
            re.compile("(?<=[Cc]ortisol, [Ss]erum 0800h is )[0-9.]* (?=nmol/L)"): 
            [[138, 635], ["Cortisol levels low (serum or plasma)", "Cortisol levels raised (serum or plasma)"], "Cortisol Serum"],
            re.compile("(?<=[Cc]ortisol, [Ss]erum 0800h: )[0-9.]* (?=nmol/L)"): 
            [[138, 635], ["Cortisol levels low (serum or plasma)", "Cortisol levels raised (serum or plasma)"], "Cortisol Serum"],
            

            re.compile("(?<=[Cc]ortisol, [Ss]erum 1600h is )[0-9.]* (?=μg/dL)"): 
            [[3, 15], ["Cortisol levels low (serum or plasma)", "Cortisol levels raised (serum or plasma)"], "Cortisol Serum"],
            re.compile("(?<=[Cc]ortisol, [Ss]erum 1600h: )[0-9.]* (?=μg/dL)"): 
            [[3, 15], ["Cortisol levels low (serum or plasma)", "Cortisol levels raised (serum or plasma)"], "Cortisol Serum"],
            re.compile("(?<=[Cc]ortisol, [Ss]erum 1600h is )[0-9.]* (?=nmol/L)"): 
            [[82, 413], ["Cortisol levels low (serum or plasma)", "Cortisol levels raised (serum or plasma)"], "Cortisol Serum"],
            re.compile("(?<=[Cc]ortisol, [Ss]erum 1600h: )[0-9.]* (?=nmol/L)"): 
            [[82, 413], ["Cortisol levels low (serum or plasma)", "Cortisol levels raised (serum or plasma)"], "Cortisol Serum"],
            
            re.compile("(?<=[Cc]ortisol, [Ss]erum 2000h is )[0-9.]* (?=μg/dL)"): 
            [[2.5, 11.5], ["Cortisol levels low (serum or plasma)", "Cortisol levels raised (serum or plasma)"], "Cortisol Serum"],
            re.compile("(?<=[Cc]ortisol, [Ss]erum 2000h: )[0-9.]* (?=μg/dL)"): 
            [[2.5, 11.5], ["Cortisol levels low (serum or plasma)", "Cortisol levels raised (serum or plasma)"], "Cortisol Serum"],
            re.compile("(?<=[Cc]ortisol, [Ss]erum 2000h is )[0-9.]* (?=nmol/L)"): 
            [[69, 317.5], ["Cortisol levels low (serum or plasma)", "Cortisol levels raised (serum or plasma)"], "Cortisol Serum"],
            re.compile("(?<=[Cc]ortisol, [Ss]erum 2000h: )[0-9.]* (?=nmol/L)"): 
            [[69, 317.5], ["Cortisol levels low (serum or plasma)", "Cortisol levels raised (serum or plasma)"], "Cortisol Serum"],
            

            re.compile("(?<=[Cc]reatine kinase, serum [Mm]ale is )[0-9.]* (?=U/L)"): 
            [[25, 90], ["Creatine kinase levels low (plasma or serum)", "Creatine kinase levels raised (plasma or serum)"], "Creatine kinase"],
            re.compile("(?<=[Cc]reatine kinase, serum [Mm]ale: )[0-9.]* (?=U/L)"): 
            [[25, 90], ["Creatine kinase levels low (plasma or serum)", "Creatine kinase levels raised (plasma or serum)"], "Creatine kinase"],
            re.compile("(?<=[Cc]reatine kinase, serum [Ff]emale is )[0-9.]* (?=U/L)"): 
            [[10, 70], ["Creatine kinase levels low (plasma or serum)", "Creatine kinase levels raised (plasma or serum)"], "Creatine kinase"],
            re.compile("(?<=[Cc]reatine kinase, serum [Ff]emale: )[0-9.]* (?=U/L)"): 
            [[10, 70], ["Creatine kinase levels low (plasma or serum)", "Creatine kinase levels raised (plasma or serum)"], "Creatine kinase"],
            

            re.compile("(?<=[Cc]reatine kinase [Mm]ale is )[0-9.]* (?=U/L)"): 
            [[25, 90], ["Creatine kinase levels low (plasma or serum)", "Creatine kinase levels raised (plasma or serum)"], "Creatine kinase"],
            re.compile("(?<=[Cc]reatine kinase [Mm]ale: )[0-9.]* (?=U/L)"): 
            [[25, 90], ["Creatine kinase levels low (plasma or serum)", "Creatine kinase levels raised (plasma or serum)"], "Creatine kinase"],
            re.compile("(?<=[Cc]reatine kinase [Ff]emale is )[0-9.]* (?=U/L)"): 
            [[10, 70], ["Creatine kinase levels low (plasma or serum)", "Creatine kinase levels raised (plasma or serum)"], "Creatine kinase"],
            re.compile("(?<=[Cc]reatine kinase [Ff]emale: )[0-9.]* (?=U/L)"): 
            [[10, 70], ["Creatine kinase levels low (plasma or serum)", "Creatine kinase levels raised (plasma or serum)"], "Creatine kinase"],
            
            re.compile("(?<=[Cc]reatine kinase is )[0-9.]* (?=U/L)"): 
            [[10, 90], ["Creatine kinase levels low (plasma or serum)", "Creatine kinase levels raised (plasma or serum)"], "Creatine kinase"],
            re.compile("(?<=[Cc]reatine kinase: )[0-9.]* (?=U/L)"): 
            [[10, 90], ["Creatine kinase levels low (plasma or serum)", "Creatine kinase levels raised (plasma or serum)"], "Creatine kinase"],
            re.compile("(?<=[Cc]reatine kinase activity is )[0-9.]* (?=U/L)"): 
            [[10, 90], ["Creatine kinase levels low (plasma or serum)", "Creatine kinase levels raised (plasma or serum)"], "Creatine kinase"],
            

            re.compile("(?<=[Cc]reatinine is )[0-9.]* (?=mg/dL)"): 
            [[0.6, 1.2], ["Creatinine levels low (plasma or serum)", "Creatinine levels raised (plasma or serum)"], "Creatinine"],
            re.compile("(?<=[Cc]reatinine: )[0-9.]* (?=mg/dL)"): 
            [[0.6, 1.2], ["Creatinine levels raised (plasma or serum)", "Creatinine levels raised (plasma or serum)"], "Creatinine"],
            re.compile("(?<=[Cc]reatinine is )[0-9.]* (?=μmol/L)"): 
            [[53, 106], ["Creatinine levels low (plasma or serum)", "Creatinine levels raised (plasma or serum)"], "Creatinine"],
            re.compile("(?<=[Cc]reatinine: )[0-9.]* (?=μmol/L)"): 
            [[53, 106], ["Creatinine levels raised (plasma or serum)", "Creatinine levels raised (plasma or serum)"], "Creatinine"],

            re.compile("(?<=[Cc]reatinine of )[0-9.]* (?=mg/dL)"): 
            [[0.6, 1.2], ["Creatinine levels low (plasma or serum)", "Creatinine levels raised (plasma or serum)"], "Creatinine"],
            re.compile("(?<=[Cc]reatinine level is: )[0-9.]* (?=mg/dL)"): 
            [[0.6, 1.2], ["Creatinine levels raised (plasma or serum)", "Creatinine levels raised (plasma or serum)"], "Creatinine"],
            re.compile("(?<=[Cc]reatinine of )[0-9.]* (?=μmol/L)"): 
            [[53, 106], ["Creatinine levels low (plasma or serum)", "Creatinine levels raised (plasma or serum)"], "Creatinine"],
            re.compile("(?<=[Cc]reatinine level is: )[0-9.]* (?=μmol/L)"): 
            [[53, 106], ["Creatinine levels raised (plasma or serum)", "Creatinine levels raised (plasma or serum)"], "Creatinine"],

            re.compile("(?<=[Ss]odium is )[0-9.]* (?=mEq/L)"): 
            [[136, 145], ["Sodium levels low (plasma or serum)", "Sodium levels raised (plasma or serum)"], "Sodium"],
            re.compile("(?<=[Ss]odium: )[0-9.]* (?=mEq/L)"): 
            [[136, 145], ["Sodium levels low (plasma or serum)", "Sodium levels raised (plasma or serum)"], "Sodium"],
            re.compile("(?<=[Ss]odium is )[0-9.]* (?=mmol/L)"): 
            [[136, 145], ["Sodium levels low (plasma or serum)", "Sodium levels raised (plasma or serum)"], "Sodium"],
            re.compile("(?<=[Ss]odium: )[0-9.]* (?=mmol/L)"): 
            [[136, 145], ["Sodium levels low (plasma or serum)", "Sodium levels raised (plasma or serum)"], "Sodium"],
            re.compile("(?<=Na\+ is )[0-9.]* (?=mEq/L)"): 
            [[136, 145], ["Sodium levels low (plasma or serum)", "Sodium levels raised (plasma or serum)"], "Sodium"],
            re.compile("(?<=Na\+: )[0-9.]* (?=mEq/L)"): 
            [[136, 145], ["Sodium levels low (plasma or serum)", "Sodium levels raised (plasma or serum)"], "Sodium"],
            re.compile("(?<=Na\+ is )[0-9.]* (?=mmol/L)"): 
            [[136, 145], ["Sodium levels low (plasma or serum)", "Sodium levels raised (plasma or serum)"], "Sodium"],
            re.compile("(?<=Na\+:)[0-9.]* (?=mmol/L)"): 
            [[136, 145], ["Sodium levels low (plasma or serum)", "Sodium levels raised (plasma or serum)"], "Sodium"],
            

            re.compile("(?<=[Pp]otassium is )[0-9.]* (?=mEq/L)"): 
            [[3.5, 5.0], ["Potassium levels low (plasma or serum)", "Potassium levels raised (plasma or serum)"], "Potassium"],
            re.compile("(?<=[Pp]otassium: )[0-9.]* (?=mEq/L)"): 
            [[3.5, 5.0], ["Potassium levels low (plasma or serum)", "Potassium levels raised (plasma or serum)"], "Potassium"],
            re.compile("(?<=[Pp]otassium is )[0-9.]* (?=mmol/L)"): 
            [[3.5, 5.0], ["Potassium levels low (plasma or serum)", "Potassium levels raised (plasma or serum)"], "Potassium"],
            re.compile("(?<=[Pp]otassium: )[0-9.]* (?=mmol/L)"): 
            [[3.5, 5.0], ["Potassium levels low (plasma or serum)", "Potassium levels raised (plasma or serum)"], "Potassium"],
            re.compile("(?<=K\+ is )[0-9.]* (?=mEq/L)"): 
            [[3.5, 5.0], ["Potassium levels low (plasma or serum)", "Potassium levels raised (plasma or serum)"], "Potassium"],
            re.compile("(?<=K\+: )[0-9.]* (?=mEq/L)"): 
            [[3.5, 5.0], ["Potassium levels low (plasma or serum)", "Potassium levels raised (plasma or serum)"], "Potassium"],
            re.compile("(?<=K\+ is )[0-9.]* (?=mmol/L)"): 
            [[3.5, 5.0], ["Potassium levels low (plasma or serum)", "Potassium levels raised (plasma or serum)"], "Potassium"],
            re.compile("(?<=K\+: )[0-9.]* (?=mmol/L)"): 
            [[3.5, 5.0], ["Potassium levels low (plasma or serum)", "Potassium levels raised (plasma or serum)"], "Potassium"],
            

            re.compile("(?<=[Cc]hloride is )[0-9.]* (?=mEq/L)"): 
            [[95, 105], ["Chloride levels low (plasma or serum)", "Chloride levels raised (plasma or serum)"], "Chloride"],
            re.compile("(?<=[Cc]hloride: )[0-9.]* (?=mEq/L)"): 
            [[95, 105], ["Chloride levels low (plasma or serum)", "Chloride levels raised (plasma or serum)"], "Chloride"],
            re.compile("(?<=[Cc]hloride is )[0-9.]* (?=mmol/L)"): 
            [[95, 105], ["Chloride levels low (plasma or serum)", "Chloride levels raised (plasma or serum)"], "Chloride"],
            re.compile("(?<=[Cc]hloride: )[0-9.]* (?=mmol/L)"): 
            [[95, 105], ["Chloride levels low (plasma or serum)", "Chloride levels raised (plasma or serum)"], "Chloride"],
            re.compile("(?<=Cl\- is )[0-9.]* (?=mEq/L)"): 
            [[95, 105], ["Chloride levels low (plasma or serum)", "Chloride levels raised (plasma or serum)"], "Chloride"],
            re.compile("(?<=Cl\-: )[0-9.]* (?=mEq/L)"): 
            [[95, 105], ["Chloride levels low (plasma or serum)", "Chloride levels raised (plasma or serum)"], "Chloride"],
            re.compile("(?<=Cl\- is )[0-9.]* (?=mmol/L)"): 
            [[95, 105], ["Chloride levels low (plasma or serum)", "Chloride levels raised (plasma or serum)"], "Chloride"],
            re.compile("(?<=Cl\-: )[0-9.]* (?=mmol/L)"): 
            [[95, 105], ["Chloride levels low (plasma or serum)", "Chloride levels raised (plasma or serum)"], "Chloride"],
            

            re.compile("(?<=[Bb]icarbonate is )[0-9.]* (?=mEq/L)"): 
            [[22, 28], ["Bicarbonate levels low (plasma)", "Bicarbonate levels raised (plasma)"], "Bicarbonate"],
            re.compile("(?<=[Bb]icarbonate: )[0-9.]* (?=mEq/L)"): 
            [[22, 28], ["Bicarbonate levels low (plasma)", "Bicarbonate levels raised (plasma)"], "Bicarbonate"],
            re.compile("(?<=[Bb]icarbonate is )[0-9.]* (?=mmol/L)"): 
            [[22, 28], ["Bicarbonate levels low (plasma)", "Bicarbonate levels raised (plasma)"], "Bicarbonate"],
            re.compile("(?<=[Bb]icarbonate: )[0-9.]* (?=mmol/L)"): 
            [[22, 28], ["Bicarbonate levels low (plasma)", "Bicarbonate levels raised (plasma)"], "Bicarbonate"],
            re.compile("(?<=HCO3\- is )[0-9.]* (?=mEq/L)"): 
            [[22, 28], ["Bicarbonate levels low (plasma)", "Bicarbonate levels raised (plasma)"], "Bicarbonate"],
            re.compile("(?<=HCO3\-: )[0-9.]* (?=mEq/L)"): 
            [[22, 28], ["Bicarbonate levels low (plasma)", "Bicarbonate levels raised (plasma)"], "Bicarbonate"],
            re.compile("(?<=HCO3\- is )[0-9.]* (?=mmol/L)"): 
            [[22, 28], ["Bicarbonate levels low (plasma)", "Bicarbonate levels raised (plasma)"], "Bicarbonate"],
            re.compile("(?<=HCO3\-: )[0-9.]* (?=mmol/L)"): 
            [[22, 28], ["Bicarbonate levels low (plasma)", "Bicarbonate levels raised (plasma)"], "Bicarbonate"],
            
            re.compile("(?<=[Mm]agnesium is )[0-9.]* (?=mEq/L)"): 
            [[1.5, 2.0], ["Magnesium levels low (plasma or serum)", "Magnesium levels raised (plasma or serum)"], "Magnesium"],
            re.compile("(?<=[Mm]agnesium: )[0-9.]* (?=mEq/L)"): 
            [[1.5, 2.0], ["Magnesium levels low (plasma or serum)", "Magnesium levels raised (plasma or serum)"], "Magnesium"],
            re.compile("(?<=[Mm]agnesium is )[0-9.]* (?=mmol/L)"): 
            [[1.5, 2.0], ["Magnesium levels low (plasma or serum)", "Magnesium levels raised (plasma or serum)"], "Magnesium"],
            re.compile("(?<=[Mm]agnesium: )[0-9.]* (?=mmol/L)"): 
            [[1.5, 2.0], ["Magnesium levels low (plasma or serum)", "Magnesium levels raised (plasma or serum)"], "Magnesium"],
            re.compile("(?<=Mg2\+ is )[0-9.]* (?=mEq/L)"): 
            [[1.5, 2.0], ["Magnesium levels low (plasma or serum)", "Magnesium levels raised (plasma or serum)"], "Magnesium"],
            re.compile("(?<=Mg2\+: )[0-9.]* (?=mEq/L)"): 
            [[1.5, 2.0], ["Magnesium levels low (plasma or serum)", "Magnesium levels raised (plasma or serum)"], "Magnesium"],
            re.compile("(?<=Mg2\+ is )[0-9.]* (?=mmol/L)"): 
            [[1.5, 2.0], ["Magnesium levels low (plasma or serum)", "Magnesium levels raised (plasma or serum)"], "Magnesium"],
            re.compile("(?<=Mg2\+: )[0-9.]* (?=mmol/L)"): 
            [[1.5, 2.0], ["Magnesium levels low (plasma or serum)", "Magnesium levels raised (plasma or serum)"], "Magnesium"],
            
            re.compile("(?<=[Ff]erritin concentration is )[0-9.]* (?=ng/mL)"): 
            [[12, 200], ["Ferritin levels low (plasma or serum)", "Ferritin levels raised (plasma or serum)"], "Ferritin"],
            re.compile("(?<=[Ff]erritin: )[0-9.]* (?=ng/mL)"): 
            [[12, 200], ["Ferritin levels low (plasma or serum)", "Ferritin levels raised (plasma or serum)"], "Ferritin"],
            re.compile("(?<=[Ff]erritin concentration is )[0-9.]* (?=μg/L)"): 
            [[12, 200], ["Ferritin levels low (plasma or serum)", "Ferritin levels raised (plasma or serum)"], "Ferritin"],
            re.compile("(?<=[Ff]erritin: )[0-9.]* (?=μg/L)"): 
            [[12, 200], ["Ferritin levels low (plasma or serum)", "Ferritin levels raised (plasma or serum)"], "Ferritin"],
            

            re.compile("(?<=[Ff]ollicle-stimulating hormone is )[0-9.]* (?=mIU/mL)"): 
            [[4, 30], ["Follicle stimulating hormone levels low (serum)", "Follicle stimulating hormone levels raised (serum)"], "Follicle Stimulating Hormone"],
            re.compile("(?<=[Ff]ollicle-stimulating hormone: )[0-9.]* (?=mIU/mL)"): 
            [[4, 30], ["Follicle stimulating hormone levels low (serum)", "Follicle stimulating hormone levels raised (serum)"], "Follicle Stimulating Hormone"],
            re.compile("(?<=[Ff]ollicle-stimulating hormone is )[0-9.]* (?=U/L)"): 
            [[4, 30], ["Follicle stimulating hormone levels low (serum)", "Follicle stimulating hormone levels raised (serum)"], "Follicle Stimulating Hormone"],
            re.compile("(?<=[Ff]ollicle-stimulating hormone: )[0-9.]* (?=U/L)"): 
            [[4, 30], ["Follicle stimulating hormone levels low (serum)", "Follicle stimulating hormone levels raised (serum)"], "Follicle Stimulating Hormone"],
            
            re.compile("(?<=pH is )[0-9.]*"): 
            [[7.35, 7.45], ["Acidosis", "Alkalosis"], "pH"],
            re.compile("(?<=pH: )[0-9.]*"): 
            [[7.35, 7.45], ["Acidosis", "Alkalosis"], "pH"],
            re.compile("(?<=pH is )[0-9.]* (?=nmol/L)"): 
            [[36, 44], ["Acidosis", "Alkalosis"], "pH"],
            re.compile("(?<=pH: )[0-9.]* (?=nmol/L)"): 
            [[36, 44], ["Acidosis", "Alkalosis"], "pH"],

            re.compile("(?<=PCO2 is )[0-9.]* (?=mm Hg)"): 
            [[33, 45], ["Acidosis, respiratory", "Alkalosis, respiratory"], "PCO2"],
            re.compile("(?<=PCO2: )[0-9.]* (?=mm Hg)"): 
            [[33, 45], ["Acidosis, respiratory", "Alkalosis, respiratory"], "PCO2"],
            re.compile("(?<=PCO2 is )[0-9.]* (?=kPa)"): 
            [[4.4, 5.9], ["Acidosis, respiratory", "Alkalosis, respiratory"], "PCO2"],
            re.compile("(?<=PCO2: )[0-9.]* (?=kPa)"): 
            [[4.4, 5.9], ["Acidosis, respiratory", "Alkalosis, respiratory"], "PCO2"],
            
            re.compile("(?<=[Gg]lucose is )[0-9.]* (?=mg/dL)"): 
            [[70, 110], ["Glucose levels low (blood, serum or plasma)", "Glucose levels raised (blood, serum or plasma)"], "Glucose Serum"],
            re.compile("(?<=[Gg]lucose: )[0-9.]* (?=mg/dL)"): 
            [[70, 110], ["Glucose levels low (blood, serum or plasma)", "Glucose levels raised (blood, serum or plasma)"], "Glucose Serum"],
            re.compile("(?<=[Gg]lucose is )[0-9.]* (?=mmol/L)"): 
            [[3.8, 6.1], ["Glucose levels low (blood, serum or plasma)", "Glucose levels raised (blood, serum or plasma)"], "Glucose Serum"],
            re.compile("(?<=[Gg]lucose: )[0-9.]* (?=mmol/L)"): 
            [[3.8, 6.1], ["Glucose levels low (blood, serum or plasma)", "Glucose levels raised (blood, serum or plasma)"], "Glucose Serum"],
            
            re.compile("(?<=[Gg]rowth [Hh]ormone is )[0-9.]* (?=ng/mL)"): 
            [[0, 5], ["Growth hormone deficiency (congenital)", None], "Growth Hormone"],
            re.compile("(?<=[Gg]rowth [Hh]ormone: )[0-9.]* (?=ng/mL)"): 
            [[0, 5], ["Growth hormone deficiency (congenital)", None], "Growth Hormone"],
            re.compile("(?<=[Gg]rowth [Hh]ormone is )[0-9.]* (?=μg/L)"): 
            [[0, 5], ["Growth hormone deficiency (congenital)", None], "Growth Hormone"],
            re.compile("(?<=[Gg]rowth [Hh]ormone: )[0-9.]* (?=μg/L)"): 
            [[0, 5], ["Growth hormone deficiency (congenital)", None], "Growth Hormone"],
            
            re.compile("(?<=IgA is )[0-9.]* (?=mg/dL)"): 
            [[76, 390], ["Immunoglobulin levels decreased (plasma or serum)", "Immunoglobulin levels raised (plasma or serum)"], "IgA"],
            re.compile("(?<=IgA: )[0-9.]* (?=mg/dL)"): 
            [[76, 390], ["Immunoglobulin levels decreased (plasma or serum)", "Immunoglobulin levels raised (plasma or serum)"], "IgA"],
            re.compile("(?<=IgA is )[0-9.]* (?=g/L)"): 
            [[0.76, 3.9], ["Immunoglobulin levels decreased (plasma or serum)", "Immunoglobulin levels raised (plasma or serum)"], "IgA"],
            re.compile("(?<=IgA: )[0-9.]* (?=g/L)"): 
            [[0.76, 3.9], ["Immunoglobulin levels decreased (plasma or serum)", "Immunoglobulin levels raised (plasma or serum)"], "IgA"],
            
            re.compile("(?<=IgE is )[0-9.]* (?=IU/mL)"): 
            [[0, 380], ["Immunoglobulin levels decreased (plasma or serum)", "Immunoglobulin levels raised (plasma or serum)"], "IgE"],
            re.compile("(?<=IgE: )[0-9.]* (?=IU/mL)"): 
            [[0, 380], ["Immunoglobulin levels decreased (plasma or serum)", "Immunoglobulin levels raised (plasma or serum)"], "IgE"],
            re.compile("(?<=IgE is )[0-9.]* (?=kIU/L)"): 
            [[0, 380], ["Immunoglobulin levels decreased (plasma or serum)", "Immunoglobulin levels raised (plasma or serum)"], "IgE"],
            re.compile("(?<=IgE: )[0-9.]* (?=kIU/L)"): 
            [[0, 380], ["Immunoglobulin levels decreased (plasma or serum)", "Immunoglobulin levels raised (plasma or serum)"], "IgE"],
            
            re.compile("(?<=IgG is )[0-9.]* (?=mg/dL)"): 
            [[650, 1500], ["Immunoglobulin levels decreased (plasma or serum)", "Immunoglobulin levels raised (plasma or serum)"], "IgG"],
            re.compile("(?<=IgG: )[0-9.]* (?=mg/dL)"): 
            [[650, 1500], ["Immunoglobulin levels decreased (plasma or serum)", "Immunoglobulin levels raised (plasma or serum)"], "IgG"],
            re.compile("(?<=IgG is )[0-9.]* (?=g/L)"): 
            [[6.5, 15], ["Immunoglobulin levels decreased (plasma or serum)", "Immunoglobulin levels raised (plasma or serum)"], "IgG"],
            re.compile("(?<=IgG: )[0-9.]* (?=g/L)"): 
            [[6.5, 15], ["Immunoglobulin levels decreased (plasma or serum)", "Immunoglobulin levels raised (plasma or serum)"], "IgG"],
            
            re.compile("(?<=IgM is )[0-9.]* (?=mg/dL)"): 
            [[40, 345], ["Immunoglobulin levels decreased (plasma or serum)", "Immunoglobulin levels raised (plasma or serum)"], "IgM"],
            re.compile("(?<=IgM: )[0-9.]* (?=mg/dL)"): 
            [[40, 345], ["Immunoglobulin levels decreased (plasma or serum)", "Immunoglobulin levels raised (plasma or serum)"], "IgM"],
            re.compile("(?<=IgM is )[0-9.]* (?=g/L)"): 
            [[0.4, 3.45], ["Immunoglobulin levels decreased (plasma or serum)", "Immunoglobulin levels raised (plasma or serum)"], "IgM"],
            re.compile("(?<=IgM: )[0-9.]* (?=g/L)"): 
            [[0.4, 3.45], ["Immunoglobulin levels decreased (plasma or serum)", "Immunoglobulin levels raised (plasma or serum)"], "IgM"],
            

            re.compile("(?<=[Ii]ron is )[0-9.]* (?=μg/dL)"): 
            [[50, 170], ["Iron deficiency", "Iron overload syndrome"], "Iron"],
            re.compile("(?<=[Ii]ron: )[0-9.]* (?=μg/dL)"): 
            [[50, 170], ["Iron deficiency", "Iron overload syndrome"], "Iron"],
            re.compile("(?<=[Ii]ron is )[0-9.]* (?=μmol/L)"): 
            [[9, 30], ["Iron deficiency", "Iron overload syndrome"], "Iron"],
            re.compile("(?<=[Ii]ron: )[0-9.]* (?=μmol/L)"): 
            [[9, 30], ["Iron deficiency", "Iron overload syndrome"], "Iron"],
            re.compile("(?<=Fe2\+ is )[0-9.]* (?=μg/dL)"): 
            [[50, 170], ["Iron deficiency", "Iron overload syndrome"], "Iron"],
            re.compile("(?<=Fe2\+: )[0-9.]* (?=μg/dL)"): 
            [[50, 170], ["Iron deficiency", "Iron overload syndrome"], "Iron"],
            re.compile("(?<=Fe2\+ is )[0-9.]* (?=μmol/L)"): 
            [[9, 30], ["Iron deficiency", "Iron overload syndrome"], "Iron"],
            re.compile("(?<=Fe2\+: )[0-9.]* (?=μmol/L)"): 
            [[9, 30], ["Iron deficiency", "Iron overload syndrome"], "Iron"],
            
            
            re.compile("(?<=[Ll]actate [Dd]ehydrogenase is )[0-9.]* (?=U/L)"): 
            [[45, 90], ["Lactate dehydrogenase levels low (plasma or serum)", "Lactate dehydrogenase levels raised (plasma or serum)"], "Lactate Dehydrogenase"],
            re.compile("(?<=[Ll]actate [Dd]ehydrogenase: )[0-9.]* (?=U/L)"): 
            [[45, 90], ["Lactate dehydrogenase levels low (plasma or serum)", "Lactate dehydrogenase levels raised (plasma or serum)"], "Lactate Dehydrogenase"],
            

            re.compile("(?<=[Ll]uteinizing [Hh]ormone is )[0-9.]* (?=mIU/mL)"): 
            [[5, 30], ["Luteinizing hormone levels low (plasma or serum)", "Luteinizing hormone levels raised (plasma or serum)"], "Luteinizing Hormone"],
            re.compile("(?<=[Ll]uteinizing [Hh]ormone: )[0-9.]* (?=mIU/mL)"): 
            [[5, 30], ["Luteinizing hormone levels low (plasma or serum)", "Luteinizing hormone levels raised (plasma or serum)"], "Luteinizing Hormone"],
            re.compile("(?<=[Ll]uteinizing [Hh]ormone is )[0-9.]* (?=U/L)"): 
            [[5, 30], ["Luteinizing hormone levels low (plasma or serum)", "Luteinizing hormone levels raised (plasma or serum)"], "Luteinizing Hormone"],
            re.compile("(?<=[Ll]uteinizing [Hh]ormone: )[0-9.]* (?=U/L)"): 
            [[5, 30], ["Luteinizing hormone levels low (plasma or serum)", "Luteinizing hormone levels raised (plasma or serum)"], "Luteinizing Hormone"],
            

            re.compile("(?<=[Oo]smolality is )[0-9.]* (?=mOsmol/kg H2O)"): 
            [[275, 295], ["Osmolality low (plasma)", "Osmolality raised (plasma)"], "Osmolality"],
            re.compile("(?<=[Oo]smolality: )[0-9.]* (?=mOsmol/kg H2O)"): 
            [[275, 295], ["Osmolality low (plasma)", "Osmolality raised (plasma)"], "Osmolality"],
            re.compile("(?<=[Oo]smolality [Ss]erum is )[0-9.]* (?=mOsmol/kg H2O)"): 
            [[275, 295], ["Osmolality low (plasma)", "Osmolality raised (plasma)"], "Osmolality"],
            re.compile("(?<=[Oo]smolality [Ss]erum: )[0-9.]* (?=mOsmol/kg H2O)"): 
            [[275, 295], ["Osmolality low (plasma)", "Osmolality raised (plasma)"], "Osmolality"],
            

            re.compile("(?<=[Pp]arathyroid [Hh]ormone is )[0-9.]* (?=pg/mL)"): 
            [[230, 630], [None, "Parathyroid hormone levels raised (plasma or serum)"], "Parathyroid Hormone"],
            re.compile("(?<=[Pp]arathyroid [Hh]ormone: )[0-9.]* (?=pg/mL)"): 
            [[230, 630], [None, "Parathyroid hormone levels raised (plasma or serum)"], "Parathyroid Hormone"],
            re.compile("(?<=[Pp]arathyroid [Hh]ormone is )[0-9.]* (?=ng/L)"): 
            [[230, 630], [None, "Parathyroid hormone levels raised (plasma or serum)"], "Parathyroid Hormone"],
            re.compile("(?<=[Pp]arathyroid [Hh]ormone: )[0-9.]* (?=ng/L)"): 
            [[230, 630], [None, "Parathyroid hormone levels raised (plasma or serum)"], "Parathyroid Hormone"],
            

            re.compile("(?<=[Aa]lkaline [Pp]hosphatase is )[0-9.]* (?=U/L)"): 
            [[20, 70], ["Alkaline phosphatase levels low (plasma or serum)", "Alkaline phosphatase levels raised (plasma or serum)"], "Alkaline Phosphatase"],
            re.compile("(?<=[Aa]lkaline [Pp]hosphatase: )[0-9.]* (?=U/L)"): 
            [[20, 70], ["Alkaline phosphatase levels low (plasma or serum)", "Alkaline phosphatase levels raised (plasma or serum)"], "Alkaline Phosphatase"],
            
            re.compile("(?<=[Tt]riglycerides is )[0-9.]* (?=mg/dL)"): 
            [[35, 160], ["Triglyceride levels low (plasma or serum)", "Triglycerides raised (plasma)"], "Triglycerides"],
            re.compile("(?<=[Tt]riglycerides: )[0-9.]* (?=mg/dL)"): 
            [[35, 160], ["Triglyceride levels low (plasma or serum)", "Triglycerides raised (plasma)"], "Triglycerides"],
            re.compile("(?<=[Tt]riglycerides is )[0-9.]* (?=mmol)"): 
            [[0.4, 1.81], ["Triglyceride levels low (plasma or serum)", "Triglycerides raised (plasma)"], "Triglycerides"],
            re.compile("(?<=[Tt]riglycerides: )[0-9.]* (?=mmol)"): 
            [[0.4, 1.81], ["Triglyceride levels low (plasma or serum)", "Triglycerides raised (plasma)"], "Triglycerides"],
            
            re.compile("(?<=[Uu]rea [Nn]itrogen is )[0-9.]* (?=mg/dL)"): 
            [[7, 18], ["Urea levels low (plasma or serum)", "Urea levels raised (plasma or serum)"], "Urea Nitrogen"],
            re.compile("(?<=[Uu]rea [Nn]itrogen: )[0-9.]* (?=mg/dL)"): 
            [[7, 18], ["Urea levels low (plasma or serum)", "Urea levels raised (plasma or serum)"], "Urea Nitrogen"],
            re.compile("(?<=[Uu]rea [Nn]itrogen of )[0-9.]* (?=mg/dL)"): 
            [[7, 18], ["Urea levels low (plasma or serum)", "Urea levels raised (plasma or serum)"], "Urea Nitrogen"],
            re.compile("(?<=[Uu]rea [Nn]itrogen is )[0-9.]* (?=mmol/L)"): 
            [[1.2, 3.0], ["Urea levels low (plasma or serum)", "Urea levels raised (plasma or serum)"], "Urea Nitrogen"],
            re.compile("(?<=[Uu]rea [Nn]itrogen: )[0-9.]* (?=mmol/L)"): 
            [[1.2, 3.0], ["Urea levels low (plasma or serum)", "Urea levels raised (plasma or serum)"], "Urea Nitrogen"],
            re.compile("(?<=[Uu]rea [Nn]itrogen of )[0-9.]* (?=mmol/L)"): 
            [[1.2, 3.0], ["Urea levels low (plasma or serum)", "Urea levels raised (plasma or serum)"], "Urea Nitrogen"],
            

            re.compile("(?<=[Uu]ric [Aa]cid is )[0-9.]* (?=mg/dL)"): 
            [[3.0, 8.2], ["Uric acid levels low (plasma or serum)", "Uric acid levels raised (plasma or serum)"], "Uric Acid"],
            re.compile("(?<=[Uu]ric [Aa]cid: )[0-9.]* (?=mg/dL)"): 
            [[3.0, 8.2], ["Uric acid levels low (plasma or serum)", "Uric acid levels raised (plasma or serum)"], "Uric Acid"],
            re.compile("(?<=[Uu]ric [Aa]cid is )[0-9.]* (?=mmol/L)"): 
            [[0.18, 0.48], ["Uric acid levels low (plasma or serum)", "Uric acid levels raised (plasma or serum)"], "Uric Acid"],
            re.compile("(?<=[Uu]ric [Aa]cid: )[0-9.]* (?=mmol/L)"): 
            [[0.18, 0.48], ["Uric acid levels low (plasma or serum)", "Uric acid levels raised (plasma or serum)"], "Uric Acid"],
            
            re.compile("(?<=[Cc]alcium is )[0-9.]* (?=mg/24h)"): 
            [[100, 300], ["Calcium levels low (urine)", "Calcium levels raised (urine)"], "Urine Calcium"],
            re.compile("(?<=[Cc]alcium: )[0-9.]* (?=mg/24h)"): 
            [[100, 300], ["Calcium levels low (urine)", "Calcium levels raised (urine)"], "Urine Calcium"],
            re.compile("(?<=[Cc]alcium is )[0-9.]* (?=mmol/24h)"): 
            [[2.5, 7.5], ["Calcium levels low (urine)", "Calcium levels raised (urine)"], "Urine Calcium"],
            re.compile("(?<=[Cc]alcium: )[0-9.]* (?=mmol/24h)"): 
            [[2.5, 7.5], ["Calcium levels low (urine)", "Calcium levels raised (urine)"], "Urine Calcium"],
            
            re.compile("(?<=[Cc]a2\+ is )[0-9.]* (?=mg/24h)"): 
            [[100, 300], ["Calcium levels low (urine)", "Calcium levels raised (urine)"], "Urine Calcium"],
            re.compile("(?<=[Cc]a2\+: )[0-9.]* (?=mg/24h)"): 
            [[100, 300], ["Calcium levels low (urine)", "Calcium levels raised (urine)"], "Urine Calcium"],
            re.compile("(?<=[Cc]a2\+ is )[0-9.]* (?=mmol/24h)"): 
            [[2.5, 7.5], ["Calcium levels low (urine)", "Calcium levels raised (urine)"], "Urine Calcium"],
            re.compile("(?<=[Cc]a2\+: )[0-9.]* (?=mmol/24h)"): 
            [[2.5, 7.5], ["Calcium levels low (urine)", "Calcium levels raised (urine)"], "Urine Calcium"],
            
            re.compile("(?<=[Ee]striol is )[0-9.]* (?=mg/24h)"): 
            [[6, 42], [None, "Estrogen excess"], "Urine Calcium"],
            re.compile("(?<=[Ee]striol: )[0-9.]* (?=mg/24h)"): 
            [[6, 42], [None, "Estrogen excess"], "Urine Calcium"],
            re.compile("(?<=[Ee]striol is )[0-9.]* (?=μmol/24h)"): 
            [[21, 146], [None, "Estrogen excess"], "Urine Calcium"],
            re.compile("(?<=[Ee]striol: )[0-9.]* (?=μmol/24h)"): 
            [[21, 146], [None, "Estrogen excess"], "Urine Calcium"],


            }


general_normal = {"is normal", "are normal", "within normal range", "within normal limits", "no abnormalities", "no other abnormalities"}
# general_abnormal = {"is abnormal", "are abnormal", "less than normal", "lower than normal", }
general_high = {"higher than normal", "higher", "elevated", "high"}

general_low = {"lower than normal", " lower ", " low "}

all_tests = {re.compile("[Aa]lanine [Aa]minotransferase"): [None, "ALT raised"],
             re.compile("ALT"): [None, "ALT raised"],
             re.compile("[Aa]mylase"): ["Amylase levels low (plasma or serum)", "Amylase levels raised (plasma or serum)"],
             re.compile("[Aa]spartate [Aa]minotransferase"): [None, "AST raised"],
             re.compile("AST"): [None, "AST raised"],
             re.compile("[Bb]ilirubin"): [None, "Bilirubin levels raised (plasma or serum)"],
             re.compile("[Cc]alcium"): ["Calcium levels low (plasma)", "Calcium levels raised (plasma)"],
             re.compile("Ca2\+"): ["Calcium levels low (plasma)", "Calcium levels raised (plasma)"],
             re.compile("[Cc]holesterol"): ["Cholesterol levels low (plasma or serum)", "Cholesterol levels raised (plasma or serum)"],
             re.compile("[Cc]ortisol"): ["Cortisol levels low (serum or plasma)", "Cortisol levels raised (serum or plasma)"],
             re.compile("[Cc]reatine [Kk]inase"): ["Creatine kinase levels low (plasma or serum)", "Creatine kinase levels raised (plasma or serum)"],
             re.compile("[Ss]odium"): ["Sodium levels low (plasma or serum)", "Sodium levels raised (plasma or serum)"],
             re.compile("Na\+"): ["Sodium levels low (plasma or serum)", "Sodium levels raised (plasma or serum)"],
             re.compile("[Pp]otassium"): ["Potassium levels low (plasma or serum)", "Potassium levels raised (plasma or serum)"],
             re.compile("K\+"): ["Potassium levels low (plasma or serum)", "Potassium levels raised (plasma or serum)"],
             re.compile("[Cc]hloride"): ["Chloride levels low (plasma or serum)", "Chloride levels raised (plasma or serum)"],
             re.compile("Cl\-"): ["Chloride levels low (plasma or serum)", "Chloride levels raised (plasma or serum)"],
             re.compile("[Bb]icarbonate"): ["Bicarbonate levels low (plasma)", "Bicarbonate levels raised (plasma)"],
             re.compile("HCO3\-"): ["Bicarbonate levels low (plasma)", "Bicarbonate levels raised (plasma)"],
             re.compile("[Mm]agnesium"): ["Magnesium levels low (plasma or serum)", "Magnesium levels raised (plasma or serum)"],
             re.compile("Mg2\+"): ["Magnesium levels low (plasma or serum)", "Magnesium levels raised (plasma or serum)"],
             re.compile("[Ff]erritin"): ["Ferritin levels low (plasma or serum)", "Ferritin levels raised (plasma or serum)"],
             re.compile("[Ff]ollicle-stimulating [Hh]ormone"): ["Follicle stimulating hormone levels low (serum)", "Follicle stimulating hormone levels raised (serum)"],
             re.compile("pH"): ["Acidosis", "Alkalosis"],
             re.compile("PCO2"): ["Acidosis, respiratory", "Alkalosis, respiratory"],
             re.compile("[Gg]lucose"): ["Glucose levels low (blood, serum or plasma)", "Glucose levels raised (blood, serum or plasma)"],
             re.compile("[Gg]rowth [Hh]ormone"): ["Growth hormone deficiency (congenital)", None],
             re.compile("IgA"): ["Immunoglobulin levels decreased (plasma or serum)", "Immunoglobulin levels raised (plasma or serum)"],
             re.compile("IgE"): ["Immunoglobulin levels decreased (plasma or serum)", "Immunoglobulin levels raised (plasma or serum)"],
             re.compile("IgG"): ["Immunoglobulin levels decreased (plasma or serum)", "Immunoglobulin levels raised (plasma or serum)"],
             re.compile("IgM"): ["Immunoglobulin levels decreased (plasma or serum)", "Immunoglobulin levels raised (plasma or serum)"],
             re.compile("[Ii]ron"): ["Iron deficiency", "Iron overload syndrome"],
             re.compile("Fe2\+"): ["Iron deficiency", "Iron overload syndrome"],
             re.compile("[Ll]actate [Dd]ehydrogenase"): ["Lactate dehydrogenase levels low (plasma or serum)", "Lactate dehydrogenase levels raised (plasma or serum)"],
             re.compile("[Ll]uteinizing [Hh]ormone"): ["Luteinizing hormone levels low (plasma or serum)", "Luteinizing hormone levels raised (plasma or serum)"],
             re.compile("[Oo]smolality"): ["Osmolality low (plasma)", "Osmolality raised (plasma)"],
             re.compile("[Pp]arathyroid [Hh]ormone"): [None, "Parathyroid hormone levels raised (plasma or serum)"],
             re.compile("[Aa]lkaline [Pp]hosphatase"): ["Alkaline phosphatase levels low (plasma or serum)", "Alkaline phosphatase levels raised (plasma or serum)"],
             re.compile("[Tt]riglyceride"): ["Triglyceride levels low (plasma or serum)", "Triglycerides raised (plasma)"],
             re.compile("[Uu]rea [Nn]itrogen"): ["Urea levels low (plasma or serum)", "Urea levels raised (plasma or serum)"],
             re.compile("[Uu]ric [Aa]cid"): ["Uric acid levels low (plasma or serum)", "Uric acid levels raised (plasma or serum)"],
             re.compile("[Ee]striol"): [None, "Estrogen excess"],
            }



#returns two lists, has the mapping, the second inidates negation (value: 1)
def match_lab_tests(sentence):
    return_lists = [[], []]
    for k, v in lab_test.items():
        m = k.findall(sentence)
        if m:
            if float(m[0]) < v[0][0]: #low
                return_lists[0].append(v[1][0])
                return_lists[1].append(0)
            elif float(m[0]) > v[0][1]: #high
                return_lists[0].append(v[1][1])
                return_lists[1].append(0)
            else: #normal
                return_lists[0] += v[1]
                return_lists[1] += [1, 1] #two negations
        # elif _match_lab_tests_normal(sentence):
        #     return_lists[0] += v[1]
        #     return_lists[1] += [1, 1]
    # for 
    for k in general_normal:
        # m = k.findall(sentence)
        if k in sentence:
            for kk, vv in all_tests.items():
                mm = kk.findall(sentence)
                if mm:
                    return_lists[0] += vv
                    return_lists[1] += [1, 1]
    for k in general_high:
        # m = k.findall(sentence)
        if k in sentence:
            for kk, vv in all_tests.items():
                mm = kk.findall(sentence)
                if mm:
                    return_lists[0].append(vv[1])
                    return_lists[1].append(0)
    for k in general_low:
        # m = k.findall(sentence)
        if k in sentence:
            for kk, vv in all_tests.items():
                mm = kk.findall(sentence)
                if mm:
                    return_lists[0].append(vv[0])
                    return_lists[1].append(0)
    return (return_lists)

    
