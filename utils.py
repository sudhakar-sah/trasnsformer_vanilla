import torch 
import torch.nn as nn 
import torch.nn.functional as F 



def get_device(): 
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def scaled_dot_product(q,k,v, mask=None): 
    d_k = q.size()[-1]
    scaled = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None : 
        scaled += mask 
    attention= F.softmax(scaled, dim=-1)
    values = torch.matmul(attention , v)
    return values, attention 


class PositionWiseFeedForward(nn.Module): 
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden) # 512 -> 2048 
        self.linear2 = nn.Linear(hidden, d_model) # 2048 -> 512
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        
    def forward(self, x): # 30 x 200 x 512 
        x = self.linear1(x) # 30 x 200 x 2048 
        x = self.relu(x) # 30 x 200 x 2048 
        x = self.dropout(x) # 30 x 200 x 2048  
        x = self.linear2(x) # 30 x 200 x 512  
        return x 
    

class PositionalEncoding(nn.Module): 
    
    def __init__(self, d_model, max_sequence_length): 
        super().__init__()
        self.max_sequence_length=max_sequence_length
        self.d_model = d_model 
        
    def forward(self): 
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator=torch.pow(10000, even_i/self.d_model)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        even_PE=torch.sin(position/denominator)
        odd_PE= torch.cos(position/denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE= torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE 


