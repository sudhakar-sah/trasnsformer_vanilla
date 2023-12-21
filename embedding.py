import torch 
from torch import nn 
from utils import PositionalEncoding, get_device

class SentenceEmbedding(nn.Module): 
    """creates embedding for a given sentence 

    Args:
        nn (_type_): _description_
    """
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN): 
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN=START_TOKEN
        self.END_TOKEN=END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
        self.device = get_device()
    
    def batch_tokenize(self, batch, start_token, end_token): 
        """_summary_

        Args:
            batch (_type_): _description_
            start_token (_type_): _description_
            end_token (_type_): _description_
        """
        
        def tokenize(sentence, start_token, end_token):
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
    
    def forward(self, x, start_token, end_token): # sentence 
        """_summary_

        Args:
            x (_type_): _description_
            start_token (_type_): _description_
            end_token (_type_): _description_
        """
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(self.device)
        x = self.dropout(x + pos)
        return x
