import unicodedata
import string
import re
import random
import time
import datetime
import math
import socket
hostname = socket.gethostname()
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy
from masked_cross_entropy import *
from sense_word_weight_with_char import *
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker



USE_CUDA = True 

import pickle
with open("pretrained_dict_100G_pos_conll.pkl" , "rb") as f:
    word_, sense_ = pickle.load(f)



embed_tensor = torch.zeros(len(word_), 100)
sense_tensor = torch.zeros(len(sense_), 300)
print("changing dicts")


''' words'''
input_lang.word2index = {}
input_lang.index2word = {}

Keys = list(word_.keys())

for i in range(len(word_)):
#    print(i)
    embed_tensor[i] = torch.FloatTensor(word_[Keys[i]])
    input_lang.word2index[Keys[i]] = i 
    input_lang.index2word[i] = Keys[i]


''' sense'''

input_lang_sense.word2index = {}
input_lang_sense.index2word = {}

Keys = list(sense_.keys())   

for i in range(len(sense_)):
    
    
    sense_tensor[i] = torch.FloatTensor(sense_[Keys[i]])
    input_lang_sense.word2index[Keys[i]] = i 
    input_lang_sense.index2word[i] = Keys[i]    


# Initialize models

crf = CRF(output_lang.n_words)
#crf.load_state_dict(torch.load('best_crf_54'))
encoder = EncoderRNN(len(input_lang.index2word), len(char2index), embed_size, hidden_size, output_lang.n_words,len(input_lang_sense.index2word), len(suffix2index), 2100, crf, n_layers, dropout=dropout, if_highway = False)
encoder.load_embeddings(embed_tensor, sense_tensor)

print(encoder)

#encoder.load_state_dict(torch.load('best_model_54'))


# Initialize optimizers and criterion
#encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=1e-5)
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, momentum=0.9)



criterion = nn.CrossEntropyLoss()

# Move models to GPU
if USE_CUDA:
    torch.cuda.set_device(0)
    encoder.cuda()
    
#print(encoder)
# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def evaluate_randomly():
    [input_sentence, target_sentence] = random.choice(pairs)
    if (len(input_sentence)) > 0:
        evaluate_and_show_attention(input_sentence, target_sentence)

# Begin!
ecs = []
dcs = []
eca = 0
dca = 0

F1_score = []
best_F1_micro = -1
best_F1_macro = -1
best_F1_weighted = -1
best_acc = -1

def optimized_evaluate(test_pairs,check):
    #acc
    encoder.train(False)
    global best_F1_micro, best_F1_macro, best_F1_weighted, best_acc
    print("best F1(micro):   " , best_F1_micro,"\nbest F1(macro):   " , best_F1_macro,"\nbest F1(weighted):   " , best_F1_weighted,"\nbest Accuracy:   " , best_acc)
    length = len(test_pairs)
    right_num = 0
    all_num = 0
    accuracy = 0.0
    #F1 score
    y_true = []
    y_pred = []

    length = len(test_pairs)
    
    word2idex = vocab_for_F1()
    
    if check == 0:
        indexes = np.random.choice(length, length, replace  = False)
    else:
        indexes = np.random.choice(length, length, replace = False)
    tot_correct = 0
    tot = 0
    W = [0]*48
    for j in range(len(indexes)):    
        if j % 500 == 0:
            print("evaluating this one  : ", j)
        k = j
        [input_sentence, target_sentence, input_sentence_sense] = test_pairs[k]

        target = target_sentence.split()
        actual = input_sentence.split()
        
        '''
        ### suffixes ###
        suffix_seqs = []
        for x in actual:
            suffix_seqs.append(word2suffix[x])
#        suffix_seqs.append(suffix2index['<uns>'])
        suffix_seqs =[suffix_seqs]
        
        '''
        
        '''
        ### prefixes ###
        
        prefix_seqs = []
        for x in actual:
            prefix_seqs.append(word2prefix[x])
#        prefix_seqs.append(prefix2index['<unp>'])
        prefix_seqs =[prefix_seqs]
        
        '''
        
        '''
        ### previous way of doing spellings ###
        spelling_seqs = []
        splitted = actual
        length  = len(splitted)
        for i in range(length):
            q = return_id(splitted[i],i,length)
            spelling_seqs.append(q)
        spelling_seqs = [spelling_seqs]
        
        '''
        
        input_seqs = [indexes_from_sentence(input_lang, input_sentence)]
        input_lengths = [len(input_seqs[0])]
        input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)
#        input_batches_suffix = Variable(torch.LongTensor(suffix_seqs), volatile=True).transpose(0, 1)
#        input_batches_spelling = Variable(torch.LongTensor(spelling_seqs), volatile=True).transpose(0, 1)
#        input_batches_prefix = Variable(torch.LongTensor(prefix_seqs), volatile=True).transpose(0, 1)
  
        target_seqs = [indexes_from_sentence(output_lang, target_sentence)]
        target_batches = Variable(torch.LongTensor(target_seqs), volatile=True).transpose(0,1) 
        
        input_seqs_sense = [indexes_from_sentence(input_lang_sense, input_sentence_sense)]
        input_batches_sense = Variable(torch.LongTensor(input_seqs_sense), volatile=True).transpose(0, 1)
        
        ''' new way of doing spellings '''
        #### spellings
        spellings = []
        for i in range(input_batches.size(0)):  # on sequence
            bat = []
            for j in range(input_batches.size(1)): # on batch
                bat.append(return_mask(input_lang.index2word[int(input_batches[i][j].data.cpu().numpy()) ], i ,input_lengths[j]))
            spellings.append(bat)

        spellings = Variable(torch.FloatTensor(spellings))
        ##### last two
        last_two = []
        for i in range(input_batches.size(0)):  # on sequence
            bat = []
            for j in range(input_batches.size(1)): # on batch
                bat.append(return_lastTwo_mask(input_lang.index2word[int(input_batches[i][j].data.cpu().numpy()) ]))
            last_two.append(bat)

        last_two = Variable(torch.FloatTensor(last_two))
        
        ### suffix
        suffix = []
        for i in range(input_batches.size(0)):  # on sequence
            bat = []
            for j in range(input_batches.size(1)): # on batch
                bat.append(word2suffix[input_lang.index2word[int(input_batches[i][j].data.cpu().numpy()) ]])
            suffix.append(bat)

        suffix = Variable(torch.FloatTensor(suffix))
        
        
        if USE_CUDA:
            input_batches.cuda()
            input_batches_sense.cuda()
            spellings.cuda()
            last_two.cuda()
            suffix.cuda()
#            input_batches_suffix = input_batches_suffix.cuda()
#            input_batches_prefix = input_batches_prefix.cuda()
           
            
#        temp_left = []
#        temp_right_generated = []
#        temp_right_actual = []
        if (input_lengths[0]) > 0:
            print(input_batches)
            aas
            index = encoder(input_batches, input_lengths, input_batches_sense, None, suffix, input_batches_spelling, last_two,  None, None, train = False)
            index = index.transpose(0,1)
            
#            print(target_batches)
#            print(index)
#            print(target_batches[5].data.cpu().numpy()[0])
#            print(index[5].data.cpu().numpy()[0])
#            
#            aaa
#            a,index = predict.squeeze(1).topk(1)
            for i in range(min(index.size(0), target_batches.size(0))):
                y_pred.append(index[i].data.cpu().numpy()[0])
                y_true.append(target_batches[i].data.cpu().numpy()[0])

#                W[output_lang.word2index[target[i]]] +=1 
#                if index[i].data.cpu().numpy()[0] == output_lang.word2index[target[i]]:
#                    tot_correct = tot_correct + 1
#                tot = tot + 1
#                temp_left.append(input_lang.index2word[input_seqs[0][i]])
#                temp_right_generated.append(output_lang.index2word[index[i].data.cpu().numpy()[0]])
#                temp_right_actual.append(output_lang.index2word[target_batches[i].data.cpu().numpy()[0]])
##        
#        
#        with open("test_actual.txt",'a') as newtest:
#            newtest.write(" ".join(temp_left) + "\t" + " ".join(temp_right_actual) + "\n")              
#        newtest.close()
#        
#        with open("test_generated.txt","a") as ff:
#            ff.write(" ".join(temp_left) + "\t" + " ".join(temp_right_generated) + "\n")
#        ff.close()
        
        # Set to not-training mode to disable dropout
        
#    print("Our Accuracy: ",tot_correct/tot)
#    print(W)
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    
    f1_weighted = f1_score(y_true,y_pred,average='weighted')
    f1_micro = f1_score(y_true,y_pred,average='micro')
    f1_macro = f1_score(y_true,y_pred,average='macro')
    acc = accuracy_score(y_true,y_pred,W)

        
    print('F1-screo(micro): ', f1_micro)
    print("F1-core(macro):",f1_macro)
    print("F1-core(weighted):",f1_weighted)
    print("sklearn accuray: ",acc)
    
    
    
    F1_score.append([f1_macro,f1_micro,f1_weighted,acc])
    
    if best_F1_weighted < f1_weighted:
        best_F1_weighted = f1_weighted
    
    if best_F1_macro < f1_macro:
        best_F1_macro = f1_macro
        
    if best_F1_micro < f1_micro:
        best_F1_micro = f1_micro
              
    if best_acc < acc and  check == 0:
#        print("model saved")
        import pickle
        with open("Y's_58.pkl" , "wb") as f:
            pickle.dump([y_true,y_pred],f)
        
        torch.save(encoder.state_dict(),'best_model')
        torch.save(crf.state_dict(),'best_crf')
        best_acc = acc
    
    encoder.train(True)
    return None


LR = Variable(torch.FloatTensor([0.1]).cuda(), requires_grad = False)
#LR = learning_rate 
def train(input_batches, input_lengths, target_batches, target_lengths, input_batches_sense, spellings, last_two, suffix,  encoder,   encoder_optimizer,   criterion, max_length=5000):
#    print("training")
    # Zero gradients of both optimizers
    
    encoder_optimizer.zero_grad()
    try:
        encoder.W1.grad.data.zero_()
        encoder.W2.grad.data.zero_() 
        encoder.W3.grad.data.zero_() 
        encoder.W4.grad.data.zero_() 
        
    except Exception as e:
        pass
    
    loss = 0 # Added onto for each word

    # Run words through encoder
    lens_s = Variable(torch.LongTensor(input_lengths).cuda(), requires_grad= True)
    
    loglik = encoder(input_batches, input_lengths, input_batches_sense, target_batches, suffix, spellings, last_two)

    loss = -loglik.mean()
    
    nll_v = float(-(loglik / lens_s.float()).data[0])
    print("loss: ", float(loss.data.cpu().numpy()))
    loss.backward()
    encoder_optimizer.step()
    # Prepare input and output variables
    try:
        encoder.W1 = encoder.W1.sub(LR * encoder.W1.grad)
        encoder.W2 = encoder.W2.sub(LR * encoder.W2.grad)
        encoder.W3 = encoder.W3.sub(LR * encoder.W3.grad)
        encoder.W4 = encoder.W4.sub(LR * encoder.W4.grad)
#        pass
                
    except Exception as e:
        pass
    # Loss calculation and backpropagation
#    loss = masked_cross_entropy(
#        predict.transpose(0, 1).contiguous(), # -> batch x seq
#        target_batches.transpose(0, 1).contiguous(), # -> batch x seq
#        target_lengths
#    )
#    print("loss: ", loss.data.cpu().numpy()[0])
    
#    loss.backward()
    
    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    

    # Update parameters with optimizers
#    encoder_optimizer.step()
    
    
    return loss.data[0], ec


def vocab_for_F1():   # returns a dict of indexing for F1 score
    word2index_f1 = {}

    keys_input = input_lang.word2index.keys()
    set_vocab  = set()
    
    for k in keys_input:
        set_vocab.add(k)
    
    set_vocab.add('PAD')
    set_vocab.add('SOS')
    set_vocab.add('EOS')
    
    keys_output = output_lang.word2index.keys()
    for k in keys_output:
        set_vocab.add(k)
    
    i = 0
    set_vocab = list(set_vocab)
    for x in set_vocab:
        word2index_f1[x] = i
        i = i + 1
    return word2index_f1

#with open("pair_data.pkl","rb") as f:
#    test_pairs,pairs = pickle.load(f)
    


for i in range(len(pairs)):
    pairs[i].append(pairs_sense[i][0])
    
for i in range(len(test_pairs)):
    test_pairs[i].append(test_pairs_sense[i][0])




def return_lastTwo_mask(word):
    one_hot_list = [0] * len(char2index)
    if len(word) == 1:
        one_hot_list[char2index[word[-1]]] = 1
    else:
        one_hot_list[char2index[word[-2]]] = 1
        one_hot_list[char2index[word[-1]]] = 1
    return deepcopy(one_hot_list)

def return_all_mask(Word):
    one_hot_list = []
    for size in range(2,6):
        hot_list = [0]*len(char2index)
        if len(Word) >= size :
            for i in range(size):
                hot_list[char2index[Word[-i]]] = 1
        one_hot_list += hot_list
    return one_hot_list


if len(pairs) % batch_size !=0:
    no_of_batches = len(pairs) // batch_size + 1
    n_epochs = no_of_batches + 1
else:
    no_of_batches = len(pairs) // batch_size
    n_epochs = no_of_batches

   

while epoch < n_epochs:
    print("....." , epoch)
    epoch += 1
    print("dropout 0.5 and 0.2, w 123, no bigram, layer (1), BIN suf aagerta WSJ + CONLL")
    total_loss = 0
    for index in range(n_epochs):
        print("............................." , epoch, " /:/ ", index, " th batch")
    # Get training data for this cycle
        input_batches, input_lengths, target_batches, target_lengths, input_batches_sense = random_batch(batch_size, pairs, index)
        
        spellings = []
        for i in range(input_batches.size(0)):  # on sequence
            bat = []
            for j in range(input_batches.size(1)): # on batch
                #print(input_batches[i][j])
                word = input_lang.index2word[int(input_batches[i][j].data.cpu().numpy())]
                index = i
                sen_len = input_lengths[j]
                mask = return_mask(word, index , sen_len)
                bat.append(mask)
            spellings.append(bat)

        spellings = Variable(torch.FloatTensor(spellings).cuda(), requires_grad= True)        
        ##last 2 characters 35x10x82
        last_two = []
        for i in range(input_batches.size(0)):  # on sequence
            bat = []
            for j in range(input_batches.size(1)): # on batch
                bat.append(return_lastTwo_mask(input_lang.index2word[int(input_batches[i][j].data.cpu().numpy()) ]))
            last_two.append(bat)

        last_two = Variable(torch.FloatTensor(last_two).cuda(), requires_grad= True)
        
        ### suffix
        suffix = []
        for i in range(input_batches.size(0)):  # on sequence
            bat = []
            for j in range(input_batches.size(1)): # on batch
                bat.append(word2suffix[input_lang.index2word[int(input_batches[i][j].data.cpu().numpy()) ]])
            suffix.append(bat)

        suffix = Variable(torch.FloatTensor(suffix).cuda(), requires_grad= True)

        # Run the train function
        loss, ec = train(
            input_batches,  input_lengths, target_batches, target_lengths, input_batches_sense, spellings, last_two, suffix,
            encoder,  
            encoder_optimizer,   criterion
        )
        total_loss += loss
    #    # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss
        eca += ec
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
            print(print_summary)
            
#        if epoch % evaluate_every == 0:

#        if index % evaluate_every == 0 and index >= 100:
#    #        evaluate_randomly()
#            print("W1 = ", encoder.W1, " W2 = ", encoder.W2)
#            optimized_evaluate(test_pairs,0)
#            with open("result_original.pkl","wb") as f:
#                pickle.dump([F1_score,ACC_seq],f)
        print(encoder.gate.weight.grad)
        
    optimized_evaluate(test_pairs,0)   
    for param_group in encoder_optimizer.param_groups:
        param_group['lr'] = learning_rate / (1 + epoch * lr_decay)
        
    print("Mean Loss: ", total_loss/no_of_batches)
    
#    print("W1 = ", encoder.W1, " W2 = ", encoder.W2)
    print("W1 = ", encoder.W1, " W2 = ", encoder.W2, " W3 = ", encoder.W3, " W4 = ", encoder.W4)
        
    with open("result_original.pkl","wb") as f:
        pickle.dump([F1_score, ACC_seq],f)
    np.random.shuffle(pairs)
   
    
encoder.load_state_dict(torch.load('best_model_54'))
crf.load_state_dict(torch.load('best_crf_54'))

optimized_evaluate(test_pairs, 1)

