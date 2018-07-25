#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 15:28:25 2018

@author: bob
"""

def change_numbers(filename,token="<number>"):
    with open(filename,"r") as f:
        Data = f.readlines()
    
    return Data

Data = change_numbers("train.txt")