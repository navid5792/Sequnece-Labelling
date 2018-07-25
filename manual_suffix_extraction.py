#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 16:13:37 2018

@author: bob
"""

import numpy as np

word2pos = dict()
for i in range(len(pairs)):
    sten = pairs[i][0].split()
    pos = pairs[i][1].split()
    for j in range(len(sten)):
        if sten[j] not in word2pos:
            word2pos[sten[j]] = set()
        #if sten[j] =='the' and pos[j] =='CD':
        #    print(sten,pos)
        word2pos[sten[j]].add(pos[j])


suffix = sorted(list(suffix2index.keys()),key=lambda x : len(x),reverse=True)

possibel_pos_suffix = dict()
tup_pos = []
for i in range(len(pairs)):
    sten = pairs[i][0].split()
    pos = pairs[i][1].split()
    print(i)
    for j in range(len(sten)):
        tup_pos.append((sten[j],pos[j]))
        for suf in suffix:
            if sten[j].endswith(suf):
                if suf not in possibel_pos_suffix:
                    possibel_pos_suffix[suf] = []
                    possibel_pos_suffix[suf].append([pos[j],1])
                else:
                    flag = 0
                    for x in possibel_pos_suffix[suf]:
                        if x[0] == pos[j]:
                            flag = 1
                            x[1] +=1
                    if flag == 0:
                        possibel_pos_suffix[suf].append([pos[j],1])
                break


with open("manual_suff.txt","w") as f:
    suf = possibel_pos_suffix.keys()
    for x in suf:
        f.write("\nsuffix --> {}".format(x)+"\n\n")
        for y in possibel_pos_suffix[x]:
            f.write("pos {} ---> {} samples".format(y[0],y[1]) + "\n")
    f.write("\n")


#for suf in suffix:
#    set_pos = set()
#    print("doing it for ",suf)
#    for i in range(len(pairs)):
#        sten = pairs[i][0].split()
#        pos = pairs[i][1].split()
#        for j in range(len(sten)):
#            if sten[j].endswith(suf):
#                set_pos.add(pos[j])
#                #break
#    
#    possibel_pos_suffix[suf] = set_pos
