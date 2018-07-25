
import torch
import pickle
from copy import deepcopy
from Vocab import *
from glove import *

word2vector= {}
keys = list(input_lang.word2index.keys())

#i=0
#for line in open("./embedding/glove.6B.100d.txt", 'r'):
#    print(i)
#    i = i + 1
#    line = line.strip().split(' ')
#    vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))
#    print(vector)
#    aasas
#    if line[0] in keys:
#        word2vector[line[0]] = vector

glo = Glove_parser('embedding/glove.6B.100d.txt', 100)
word2vec = glo.get_dict()
count= 0
for i in range(len(keys)):
    print(i)
    if keys[i].lower() in word2vec:
        word2vector[keys[i]] = word2vec[keys[i].lower()]
    else:
        count += 1

print(count)
with open("sense_dictionary_pos+conll.pkl" , "rb") as f:
    sense_dict = pickle.load(f)
    
keys = list(sense_dict.keys())
for i in range(len(keys)):
    sense_dict[keys[i]] = [float(i) for i in list(sense_dict[keys[i]].reshape(300))]

for x in input_lang.word2index.keys():
    if x not in word2vector.keys():
        print(x + "\n")
        word2vector[x] = torch.randn(100)


with open("pretrained_dict_100G_pos_conll.pkl" , "wb") as f:
    pickle.dump([word2vector,sense_dict],f)


#import torch
#import pickle
#from copy import deepcopy
#import numpy as np
#from Vocab import *
#from gensim.models import KeyedVectors
#vocab = list(input_lang.word2index.keys())
#word_vectors = KeyedVectors.load_word2vec_format('./embedding/GoogleNews-vectors-negative300.bin', binary=True)
#word2vector= {}
#keys = list(input_lang.word2index.keys())
#count = 0
#for i in range (len(keys)):
#    try:
#        word2vector[keys[i]] = torch.FloatTensor(word_vectors.get_vector(keys[i])).unsqueeze(1)
#    except KeyError:
#        word2vector[keys[i]] = torch.randn(300,1) 
#        count = count + 1
#
#with open("sense_dictionary.pkl" , "rb") as f:
#    sense_dict = pickle.load(f)
#    
#keys = list(sense_dict.keys())
#for i in range(len(keys)):
#    sense_dict[keys[i]] = [float(i) for i in list(sense_dict[keys[i]].reshape(300))]
#
##for x in input_lang.word2index.keys():
##    if x not in word2vector.keys() and x not in sense_dict.keys():
##        print(x + "\n")
##        word2vector[x] = torch.randn(300,1)
##
##
#with open("pretrained_dict_word2vec.pkl" , "wb") as f:
#    pickle.dump([word2vector,sense_dict],f)

