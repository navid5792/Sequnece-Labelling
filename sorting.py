#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:15:55 2018

@author: bob
"""

import numpy as np
import pickle
from copy import deepcopy
with open("baal.pkl","rb") as f:
    Data,Len = pickle.load(f)

def sort_sentence(Sent,Real_len):
    
    length = len(Sent)
    index = [x for x in range(length)]
    L = []
    for i in range(len(Real_len)):
        L.append((Real_len[i],i))
    
    #print(L)
    L = sorted(L,reverse = True)
    S = []
    for i in range(len(L)):
        S.append(Sent[L[i][1]])

        
    return S,L


before_sort = deepcopy(Data)

S,baal  = sort_sentence(deepcopy(Data[0]),deepcopy(Len[0]))



#b = make_it_real(deepcopy(a),deepcopy(baal))
