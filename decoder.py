import torch 
import torch.nn as nn 
from layer_norm import LayerNormalization
from attention import MultiHeadAttention, MultiHeadCrossAttention
from utils import PositionWiseFeedForward
from embedding import SentenceEmbedding

class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob): 
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention= MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)
        
    def forward(self, x, y, self_attention_mask, cross_attention_mask): 
        y_residue = y 
        y = self.self_attention(y, mask=self_attention_mask)  
        y = self.dropout1(y)
        y = self.norm1(y + y_residue)
        
        y_residue = y
        y = self.encoder_decoder_attention(x, y, mask=cross_attention_mask) 
        y = self.dropout2(y)
        y = self.norm2(y + y_residue)
        
        y_residue = y
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.norm3(y + y_residue)
        return y 

class SequentialDecoder(nn.Sequential): 
    def forward(self, *inputs): 
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values(): 
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y 

class Decoder(nn.Module): 
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
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN,END_TOKEN,PADDING_TOKEN)
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) 
                                          for _ in range(num_layers)])
        
    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token): 
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y 
    