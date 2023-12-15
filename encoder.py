

import torch 
from torch import nn 
from attention import MultiHeadAttention
from layer_norm import LayerNormalization 
from utils import PositionWiseFeedForward
from embedding import SequenceEmbedding

class EncoderLayer(nn.Module): 
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob,
): 
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameter_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameter_shape=[d_model])
        self.dropout2= nn.Dropout(p=drop_prob)
        
    def forward(self, x): 
        residual_x = x # 20 x 200 x 512 
        x = self.attention(x, mask=None) # 30 x 200 x 512  
        x = self.dropout1(x) # 30 x 200 x 512  
        x = self.norm1(x + residual_x) # 30 x 200 x 512  
        residual_x = x  # 30 x 200 x 512  
        x - self.ffn(x) # 30 x 200 x 512  
        x - self.dropout2(x) # 30 x 200 x 512  
        x = self.norm2(x + residual_x) # 30 x 200 x 512  
        return x
    
class SequentialEncoder(nn.Sequential): 
    def forward(self, *inputs): 
        x, self_attention_mask = inputs 
        for module in self._module.values(): 
            x = module(x, self_attention_mask)
        return x 
    
        
class Encoder(nn.Module): 
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length, 
                 language_to_index, 
                 START_TOKEN, 
                 END_TOKEN, 
                 PADDING_TOKEN): 
        super().__init__()
        self.sentence_embedding = SequenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN,END_TOKEN,PADDING_TOKEN)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) 
                                      for _ in range(num_layers)])
        
    def forward(self, x, self_attention_mask, start_token, end_token): 
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x

