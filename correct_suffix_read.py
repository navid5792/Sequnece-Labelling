#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 16:15:58 2018

@author: bob
"""

with open("Correct_suffixes.txt", "r") as f:
    line = f.readlines()


no_suf = {}
for i in range(len(line)):
    if line[i][0] == '-':
#        print("1",line[i])
        suffix = line[i][1:].strip().split(" ")
        #print(suffix[0])
        a = []
        for j in range(i+1,len(line)):
            if line[j][0] =='-':
                #i = j-1
                break
            val = line[j].strip().split(" ")
            
            if val[0].endswith(suffix[0]):
                a.append(val[0])
        #break
#        print(suffix[0],a)
        no_suf[suffix[0]] = a
        #dicts[lines[i][1:].strip()] = a