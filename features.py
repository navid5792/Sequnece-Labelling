'''
1 ---> non letter
2 ---> all letter
3 -- > start with Capital
4 --- > all uppercase
5 ----> all lowercase
6 ----> all number
7 ----> mix of number and letter
8 ----> app
9 ----> 's
10 ----> punc
11 ----> The / Name
12 ---> letters only

'''
digit = ['0','1','2','3','4','5','6','7','8','9']

def all_letter(Word,index,Len):
    for x in Word:
        if str.isalpha(x) !=True and x !='.':
            return False
    return True
def non_letter(Word,index,Len):
    flag = 0
    for x in Word:
        if str.isalpha(x) == False and x!='.':
            flag = 1
    return flag

def start_with_cap(Word,index,Len):     #0
    
    return str.isupper(Word[0])

def all_up(Word,index,Len): #1
    if all_letter(Word,index,Len) == False:
        return False
    return str.isupper(Word)

def all_lower(Word,index,Len): #2
    return  str.islower(Word)

def all_number(Word,index,Len):#3
    return str.isdigit(Word)

def mix_num_letter(Word,index,Len): #4
    flag_al = 0
    flag_digit = 0
    for x in Word:
        if str.isalpha(x) == True:
            flag_al = 1
        if str.isdigit(x) == True:
            flag_digit = 1
    
    if flag_al and flag_digit :
        return True
    return False

def position_start(Word,index,Len): # 5
    if index == 0:
        return 1
    return 0

def position_end(Word,index,Len): # 6
    if index == Len - 1:
        return 1
    return 0

def position_mid(Word,index,Len): # 7
    if index != 0 and index !=Len-1:
        return 1
    return 0

def is_app(Word,index,Len): #8
    return "'" in Word

def is_punc(Word,index,Len): #9
    import string
    for x in Word:
        if x in string.punctuation and x != "'":
            return 1
    return 0

def init_cap(Word,index,Len): #10
    if str.isupper(Word[0]) and index == 0:
        return 1
    return 0
def mainly_number(Word,index,Len):
    num_digit = 0
    for x in Word:
        if str.isdigit(x):
            num_digit +=1
    fract = num_digit/(len(Word))
    if fract >0.5:
        return True
    
    return False

Functlist = [non_letter,all_letter,start_with_cap,all_up,all_lower,all_number,mix_num_letter,position_start,position_end,position_mid,is_app,is_punc,init_cap,mainly_number]

def return_id(Word,W_index,S_len):
    mask = 0
    for i in range(len(Functlist)):
        if Functlist[i](Word,W_index,S_len):
            print("True",i,Functlist[i])
            mask = mask + 2**i
    return mask

def return_mask(Word,W_index,S_len):
    mask = []
    for i in range(len(Functlist)):
        if Functlist[i](Word,W_index,S_len):
#            print("True",i,Functlist[i])
            #mask = mask + 2**i
            mask.append(1)
        else:
            mask.append(0)
            
    return mask
    
