
import torch 
from torch import nn 
from utils import scaled_dot_product

class MultiHeadCrossAttention(nn.Module): 
    """_summary_

    Args:
        nn (_type_): _description_
    """
    
    def __init__(self, d_model=None, num_heads=None): 
        super().__init__()
        self.d_model =d_model 
        self.num_heads = num_heads 
        self.head_dim = d_model // num_heads 
        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.q_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model) 
        
        
    def forward(self, x, y, mask=None): 
        batch_size, sequence_length, d_model = x.size() 
        kv=self.kv_layer(x) 
        q = self.q_layer(y) 
        kv=kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim) 
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim) 
        kv = kv.permute(0,2,1,3)  
        q = q.permute(0,2,1,3) 
        k,v = kv.chunk(2, dim=-1)       

        values, attention = scaled_dot_product(q,k,v, mask) # no mask for cross attention 
        values = values.permute(0,2,1,3).reshape(batch_size, sequence_length, self.num_heads*self.head_dim) 
        out = self.linear_layer(values) 
        return out
    
    
class MultiHeadAttention(nn.Module): 
    
    def __init__(self, d_model, num_heads): 
        super().__init__()
        self.d_model =d_model 
        self.num_heads = num_heads 
        self.head_dim = d_model // num_heads 
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model) 
        
    def forward(self, x, mask=None): 
        batch_size, sequence_length, d_model = x.size()

        qkv=self.qkv_layer(x)
        qkv=qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0,2,1,3)
        q,k,v = qkv.chunk(3, dim=-1)       
        values, attention = scaled_dot_product(q,k,v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out
