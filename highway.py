"""
.. module:: highway
    :synopsis: highway network
 
.. moduleauthor:: Liyuan Liu
"""

import torch
import torch.nn as nn
import utils

class hw(nn.Module):
    """Highway layers

    args: 
        size: input and output dimension
        dropout_ratio: dropout ratio
    """
   
    def __init__(self, size, num_layers = 1):
        super(hw, self).__init__()
        self.size = size
        self.num_layers = num_layers
        self.trans = nn.ModuleList()
        self.gate = nn.ModuleList()
        self.dropout = nn.Dropout(p=0.5)

        for i in range(num_layers):
            tmptrans = nn.Linear(size, size) #600 600
            tmpgate = nn.Linear(size, size) #600,600
            self.trans.append(tmptrans)
            self.gate.append(tmpgate)


    def rand_init(self):
        """
        random initialization
        """
        for i in range(self.num_layers):
            utils.init_linear(self.trans[i])
            utils.init_linear(self.gate[i])

    def forward(self, x):
        """
        update statics for f1 score

        args: 
            x (ins_num, hidden_dim): input tensor
        return:
            output tensor (ins_num, hidden_dim)
        """
        
        #print(x) # 9x10x600
        
        g = nn.functional.sigmoid(self.gate[0](x))
        #print(g) # 9x10x600
        h = nn.functional.relu(self.trans[0](x))
        #print(h) # 9x10x600
        x = g * h + (1 - g) * x        # gate * transformed input + (1 - gate) * actual input     
                                       # Eq. (3) of highway paper            
        #print(x) # 9x10x600
        

        for i in range(1, self.num_layers):
            #asasasas
            x = self.dropout(x)
            g = nn.functional.sigmoid(self.gate[i](x))
            h = nn.functional.relu(self.trans[i](x))
            x = g * h + (1 - g) * x

        return x
