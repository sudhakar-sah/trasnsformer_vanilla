
import torch 
import torch.nn as nn 
from encoder import Encoder
from decoder import Decoder 
from utils import get_device

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