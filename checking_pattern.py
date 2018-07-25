#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:21:11 2018

@author: bob
"""

Word = list(input_lang.word2index.keys())

#print(Word)
test_word = []
test_pos = []

for i  in range(len(test_pairs)):
    x,y = test_pairs[i]
    
    test_word += x.split()
    test_pos += y.split()


count = 0
'''
for j in range(len(test_word)):
    flag_dig = 0
    flag_cap = 0
    flag_low = 0
    x = test_word[j]
    newword = list(x)    
    for l in x:
        if str.isdigit(l):
            flag_dig = 1
        if str.islower(l):
            flag_low = 1
        if str.isupper(l):
            flag_cap = 1
    
    if flag_cap and flag_dig and flag_low:
        for i in range(len(newword)):
            if str.isdigit(newword[i]):
                newword[i] = '0'
            elif str.islower(newword[i]):
                newword[i] = 'a'
            elif str.isupper(newword[i]):
                newword[i]= 'A'
                    
        count +=1
        print(x,"".join(newword),test_pos[j])

'''

import pickle
with open("Y's.pkl", "rb") as f:
    y_true,y_pred = pickle.load(f)

print(len(y_true))

wrong =0
for j in range(len(test_word)):
    if "-" in test_word[j]:
        print(test_word[j],test_pos[j],"----", output_lang.index2word[y_true[j]], output_lang.index2word[y_pred[j]])
        if y_true[j] != y_pred[j]:
            wrong +=1
        count +=1

print(count, wrong)


