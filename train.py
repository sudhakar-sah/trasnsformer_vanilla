import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from transformer import Transfomer
from utils import get_device
import numpy as np 
from tqdm import tqdm

START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'
NEG_INFTY = -1e9

def extract_text(filename): 
    with open(filename) as f: 
        text = f.readlines()
        
    eng_sentences = [text[i].split("\t")[0][:-1] for i in range(len(text))]
    kor_sentences = [text[i].split("\t")[1][:-1] for i in range(len(text))]

    eng_dump = ""
    for i in eng_sentences: 
        eng_dump = eng_dump + " " + i

    kor_dump = ""
    for i in kor_sentences: 
        kor_dump = kor_dump + " " + i


    return eng_dump, kor_dump, eng_sentences, kor_sentences


def create_masks(eng_batch, kn_batch, max_sequence_length):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length] , True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)

    for idx in range(num_sentences):
      eng_sentence_length, kn_sentence_length = len(eng_batch[idx]), len(kn_batch[idx])
      eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
      kn_chars_to_padding_mask = np.arange(kn_sentence_length + 1, max_sequence_length)
      encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
      encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
      decoder_padding_mask_self_attention[idx, :, kn_chars_to_padding_mask] = True
      decoder_padding_mask_self_attention[idx, kn_chars_to_padding_mask, :] = True
      decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
      decoder_padding_mask_cross_attention[idx, kn_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

def create_vocab(text):
    vocab = sorted(list(set(text)))
    vocab.insert(0, START_TOKEN)
    vocab.append(PADDING_TOKEN)
    vocab.append(END_TOKEN)
    vocab_size=len(vocab)
    return vocab, vocab_size 


def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True

def is_valid_length(sentence, max_sequence_length):
    return len(list(sentence)) < (max_sequence_length - 1) # need to re-add the end token so leaving 1 space



class TextDataSet(Dataset): 
    
    def __init__(self, eng_sentences, kor_sentences): 
        self.eng_sentences = eng_sentences 
        self.kor_sentences = kor_sentences 
        
    def __len__(self): 
        return len(self.eng_sentences)
    
    def __getitem__(self, index):
        return self.eng_sentences[index], self.kor_sentences[index]
    
def main():    

    eng_dump, kor_dump, eng_sentences, kor_sentences = extract_text('/home/sudhakar/prj/my_codes/vit/data/kor.txt')
    
    kor_vocab, _ = create_vocab(kor_dump)
    eng_vocab, _ = create_vocab(eng_dump)

    index_to_eng = {k:v for k,v in enumerate(eng_vocab)}
    eng_to_index = {v:k for k,v in enumerate(eng_vocab)}
    index_to_kor = {k:v for k,v in enumerate(kor_vocab)}
    kor_to_index = {v:k for k,v in enumerate(kor_vocab)}
    
    

    d_model = 512
    batch_size = 30
    ffn_hidden = 2048
    num_heads = 8
    drop_prob = 0.1
    num_layers = 1
    max_sequence_length = 200 
    kor_vocab_size = len(kor_vocab)
    eng_vocab_size = len(eng_vocab)
    
    valid_sentence_indicies = []
    for index in range(len(kor_sentences)):
        kannada_sentence, english_sentence = kor_sentences[index], eng_sentences[index]
        if is_valid_length(kannada_sentence, max_sequence_length) \
        and is_valid_length(english_sentence, max_sequence_length) \
        and is_valid_tokens(kannada_sentence, kor_vocab):
            valid_sentence_indicies.append(index)
    
    kor_sentences = [kor_sentences[i] for i in valid_sentence_indicies]
    eng_sentences = [eng_sentences[i] for i in valid_sentence_indicies]
        


    transformer = Transfomer(d_model, 
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
                            PADDING_TOKEN)

    dataset = TextDataSet(eng_sentences=eng_sentences, kor_sentences=kor_sentences)
    train_loader = DataLoader(dataset, batch_size, shuffle=True)
    iterator = iter(train_loader)
    
    criterian = nn.CrossEntropyLoss(ignore_index=kor_to_index[PADDING_TOKEN], reduction='none')
    for params in transformer.parameters(): 
        if params.dim() > 1 : 
            nn.init.xavier_uniform_(params)
            
    optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
    device = get_device()
    
    transformer.train()
    transformer.to(device)
    total_loss = 0
    num_epochs = 50

    for epoch in range(num_epochs): 
        print (f'Epoch : {epoch}')
        iterator = iter(train_loader)
        for batch_num, batch in enumerate(tqdm(iterator)): 
            transformer.train()
            eng_batch, kor_batch = batch 
            encoder_self_attention_mask, decoder_self_attention_mask, decoder_crosss_attention_mask = create_masks(eng_batch, kor_batch, max_sequence_length)
            encoder_self_attention_mask = encoder_self_attention_mask.to(device)
            decoder_self_attention_mask = decoder_self_attention_mask.to(device)
            decoder_crosss_attention_mask = decoder_crosss_attention_mask.to(device)
            
            optim.zero_grad()
            
            kor_predictions = transformer(eng_batch,
                                        kor_batch,
                                        encoder_self_attention_mask,
                                        decoder_self_attention_mask,
                                        decoder_crosss_attention_mask,
                                        enc_start_token=False,
                                        enc_end_token=False,
                                        dec_start_token=True,
                                        dec_end_token=True)
            
            kor_labels = transformer.decoder.sentence_embedding.batch_tokenize(kor_batch, start_token=False,end_token=True)
            loss = criterian(
                kor_predictions.view(-1, kor_vocab_size).to(device),
                kor_labels.view(-1).to(device)
            ).to(device)
            
            valid_indices = torch.where(kor_labels.view(-1) == kor_to_index[PADDING_TOKEN], False, True)
            loss = loss.sum() / valid_indices.sum()
            loss.backward()
            optim.step()
            
            if batch_num % 100 == 0:
                print (f'Iteration : {batch_num} : loss : {loss.item()}')
                print (f'Engllish : {eng_batch[0]}')
                print (f'Korean   : {kor_batch[0]}')
                kor_sentence_predicted = torch.argmax(kor_predictions[0], axis=1)
                predicted_sentence = ""
                for idx in kor_sentence_predicted: 
                    if idx == kor_to_index[END_TOKEN]: 
                        break 
                    predicted_sentence += index_to_kor[idx.item()]
                print (f'Korean prediction : {predicted_sentence}')
                    
                # transformer.eval()
                # kor_sentence=("",)
                # eng_sentence = ("should we go to the mall?")
                
                # for word_counter in range(max_sequence_length): 
                        
            # 
            # print (f'kor_prediction : {torch.argmax(kor_predictions, dim=-1)}, kor_labels : {kor_labels}')
            
if __name__ == "__main__":
    main()







