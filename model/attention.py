import math
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MultiHeadAttention(nn.Module):    
    def __init__(self, heads: int, d_model: int):        
        super(MultiHeadAttention, self).__init__()

        assert d_model % heads == 0
        self.d_k    = d_model // heads
        self.heads  = heads
        self.scaled = math.sqrt(d_model)

        self.query      = nn.Linear(d_model, d_model)
        self.key        = nn.Linear(d_model, d_model)
        self.value      = nn.Linear(d_model, d_model)
        self.out        = nn.Linear(d_model, d_model)
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        query   = self.query(query)
        key     = self.key(key)        
        value   = self.value(value)   
        
        query   = query.reshape(query.size(0), -1, self.heads, self.d_k).transpose(1, 2)
        key     = key.reshape(key.size(0), -1, self.heads, self.d_k).transpose(1, 2)
        value   = value.reshape(value.size(0), -1, self.heads, self.d_k).transpose(1, 2)
       
        scores      = query @ key.transpose(2, 3)
        scores      = scores / math.sqrt(key.size(-1))
        
        if mask is not None:
            min_type_value  = torch.finfo(scores.dtype).min
            scores  = scores.masked_fill(mask == 0, min_type_value)
             
        weights     = F.softmax(scores, dim = -1)

        context     = weights @ value
        context     = context.transpose(1, 2).flatten(2)

        interacted  = self.out(context)
        return interacted