
import torch 
from torch import nn 
from utils import scaled_dot_product

class MultiHeadCrossAttention(nn.Module): 
    
    def __init__(self, d_model=None, num_heads=None): 
        super().__init__()
        self.d_model =d_model 
        self.num_heads = num_heads 
        self.head_dim = d_model // num_heads 
        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.q_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model) 
        
        
    def forward(self, x, y, mask=None): 
        batch_size, sequence_length, input_dim = x.size() # 30 x 200 x 512 
        kv=self.kv_layer(x) # 30 x 200 x 1024
        q = self.q_layer(y) # 30 x 200 x 512 
        kv=kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim) # 30 x 200 x 8 x 128
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim) # 30 x 200 x 8 x 64 
        kv = kv.permute(0,2,1,3)  # 30 x 8 x 200 x 128 
        q = q.permute(0,2,1,3) # 30 x 8 x 200 x 64 
        k,v = kv.chunk(2, dim=-1) # K : 30 x 8 x 200 x 64 , v : same        

        values, attention = scaled_dot_product(q,k,v, mask)  # 30 x 8 x 200 x 64 
        # attention 
        values = values.reshape(batch_size, sequence_length, self.num_heads*self.head_dim) # 30 x 200 x 512 
        out = self.linear_layer(values) # 30 x 200 x 512 
        return out
    
    
class MultiHeadAttention(nn.Module): 
    
    def __init__(self, input_dim, d_model, num_heads): 
        super().__init__()
        self.input_dim = input_dim 
        self.d_model =d_model 
        self.num_heads = num_heads 
        self.head_dim = d_model // num_heads 
        self.qkv_layer = nn.Linear(input_dim, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model) 
        
    def forward(self, x, mask=None): 
        batch_size, sequence_length, input_dim = x.size()

        qkv=self.qkv_layer(x)
        qkv=qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0,2,1,3)
        q,k,v = qkv.chunk(3, dim=-1)       
        values, attention = scaled_dot_product(q,k,v, mask=None)
        values = values.reshape(batch_size, sequence_length, self.num_heads*self.head_dim)
        out = self.linear_layer(values)
        return out
