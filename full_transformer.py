import numpy as np 
import torch 
import math 
from torch import nn
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



class SequenceEmbedding(nn.Module): 
    """creates embedding for a given sentence 

    Args:
        nn (_type_): _description_
    """
    def __inii__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN): 
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embeddding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN=START_TOKEN
        self.END_TOKEN=END_TOKEN
        self.device = get_device()
    
    def batch_tokenize(self, batch, start_token=True, end_token=True): 
        
        def tokenize(sentence, start_token=True, end_token=True):
            sentence_word_indices = [self.language_to_index[token] for token in list(sentence)]
            if start_token: 
                sentence_word_indices.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token : 
                sentence_word_indices.append(self.language_to_index[self.END_TOKEN])
            
            # padding 
            for _ in range(len(sentence_word_indices), self.max_sequence_length): 
                sentence_word_indices.append(self.language_to_index[self.PADDING_TOKEN])
            
            return torch.tensor(sentence_word_indices)
        
        tokenized = [] 
        for sentence_num in range(len(batch)): 
            tokenized.append(tokenize(batch[sentence_num], start_token, end_token))
        tokenized = torch.stack(tokenized)
        return tokenized.to(self.device)
    
    def forward(self, x, end_token=True): # sentence 
        x = self.batch_tokenize(x, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(self.device)
        x = self.dropout(x + pos)
        
        
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
    
class LayerNormalization(): 
    def __init__(self, parameter_shape, eps=1e-5): 
        self.parameters_shape=parameter_shape
        self.eps=eps # to avoid division by zero
        self.gamma = nn.Parameter(torch.ones(parameter_shape)) # gamma initiallized as 1 
        self.beta = nn.Parameter(torch.zeros(parameter_shape)) # beta initialized as zero 
        
    def forward(self, inputs): 
        dims = [-(i+1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs-mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std 
        out = self.gamma * y + self.beta 
        return out 
    
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


class SequenceEmbedding(nn.Module): 
    """creates embedding for a given sentence 

    Args:
        nn (_type_): _description_
    """
    def __inii__(self, 
                 max_sentence_length, 
                 d_model, 
                 language_to_index, 
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN): 
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sentence_length
        self.embeddding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, self.max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN=START_TOKEN
        self.END_TOKEN=END_TOKEN
        self.device = get_device()
    
    def batch_tokenize(self, batch, start_token=True, end_token=True): 
        
        def tokenize(sentence, start_token=True, end_token=True):
            sentence_word_indices = [self.language_to_index[token] for token in list(sentence)]
            if start_token: 
                sentence_word_indices.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token : 
                sentence_word_indices.append(self.language_to_index[self.END_TOKEN])
            
            # padding 
            for _ in range(len(sentence_word_indices), self.max_sequence_length): 
                sentence_word_indices.append(self.language_to_index[self.PADDING_TOKEN])
            
            return torch.tensor(sentence_word_indices)
        
        tokenized = [] 
        for sentence_num in range(len(batch)): 
            tokenized.append(tokenize(batch[sentence_num], start_token, end_token))
        tokenized = torch.stack(tokenized)
        return tokenized.to(self.device)
    
    def forward(self, x, start_token = False, end_token=True): # sentence 
        x = self.batch_tokenize(x, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(self.device)
        x = self.dropout(x + pos)
        

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


class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob): 
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameter_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention= MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNormalization(parameter_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNormalization(parameter_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)
        
    def forward(self, x, y, decoder_mask, _debug =False): 
        y_residue = y 
        if _debug :
            print ("Mask Self Attention")
            print ("-------------------")
        y = self.self_attention(y, mask=decoder_mask) # 30 x 200 x 512 
        y = self.dropout1(y)
        y = self.norm1(y + y_residue)
        
        y_residue = y
        y = self.encoder_decoder_attention(x, y, mask=None) # 30 x 200 x 512 
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
        self.sentence_embedding = SequenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN,END_TOKEN,PADDING_TOKEN)
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) 
                                          for _ in range(num_layers)])
        
    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token): 
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y 
    
        
        
class Transfomer(nn.Module): 
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob,
                 num_layers, 
                 max_sequence_length,
                 kor_vocab_size, 
                 eng_vocab_size, 
                 eng_to_index, 
                 kor_to_index,
                 START_TOKEN, 
                 END_TOKEN,
                 PADDING_TOKEN): 
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, eng_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, kor_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN) 
        self.linear = nn.Linear(d_model, kor_vocab_size)
        self.device = get_device() 
 
    def forward(self, 
                x, 
                y, 
                encoder_self_attention_mask=None, 
                decoder_self_attention_mask=None,
                decoder_cross_attention_mask=None,
                enc_start_token=False, 
                enc_end_token=False,
                dec_start_token=False, 
                dec_end_token=False): 
        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token) 
        out = self.linear(out)
        return out 
                
        
        