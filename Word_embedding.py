#!/usr/bin/env python
# coding: utf-8

# In[46]:


import torch
import re
import transformers
import pickle
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import gensim
from gensim.models import Word2Vec
import scipy.io as sio
from glob import glob
from tqdm import tqdm
from time import time
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoder = BertModel.from_pretrained('bert-base-uncased')
word2vec = gensim.models.Word2Vec.load(r'E:/PhD/data/Di_Liberto/word2vec_bnc/word2vec_bnc_400d.model')

def reconstruct_sents(offsets,words,boundaries):

    values = sorted(list(set(offsets + boundaries)))
    padding = [0] + [1 if val in offsets 
               else 0
              for val in values] + [0] # add one more zero at the end in case that the list ends with 1; add one more zero at beginning in case that the list starts with 1
    
    # retrieve indices of valid values
    valid_idx = []
    for i in range(len(values)-1): # to avoid out of range
        curr_value = values[i]
        # if the current padding is one and the next is zero
        if padding[i+1] == 1 and padding[i+2] == 0:
            valid_idx.append(offsets.index(curr_value))
     
    valid_idx += [offsets.index(val) for val in offsets if val in boundaries]
    valid_idx = sorted(list(set(valid_idx)))
    offset_sent_final = [offsets[idx] for idx in valid_idx]
    
    init_id = 0
    sent_list = []
    for idx in valid_idx:
        sent_list.append(words[init_id:idx+1])
        init_id = idx+1
    
    return offset_sent_final, sent_list

def process_stim(data):
    
    boundaries = [round(float(val),2) for val in data['sentence_boundaries'][0]]
    onsets = [float(val[0]) for val in data['onset_time']]
    offsets = [float(val[0]) for val in data['offset_time']]
    words =[str(val[0][0]).strip() for val in data['wordVec']]
    
    offsets_sf, sent_list = reconstruct_sents(offsets, words, boundaries)
    
    return offsets_sf, sent_list, boundaries, onsets

def word2vec_vals(sent_list):
    
    dissi_vals = []
    
    for i in range(len(sent_list)):
        if i == 0:
            for n in range(len(sent_list[0])):
                if n == 0:
                    dissi_vals.append(0)
                else:
                    target_vec = word2vec.wv[sent_list[i][n]]
                    vec_list = [word2vec.wv[w] for w in sent_list[i][:n]]
                    averaged_vec = sum(vec_list) / len(vec_list)
                    #corr = cosine(target_vec,averaged_vec)
                    corr,_ = pearsonr(target_vec,averaged_vec)
                    dissi_vals.append(1-corr)
        
        else:        
            for n in range(len(sent_list[i])):
                target_vec = word2vec.wv[sent_list[i][n]]
                vec_list = [word2vec.wv[w] for w in sent_list[i-1]] if n == 0 else [word2vec.wv[w] for w in sent_list[i][:n]]
                averaged_vec = sum(vec_list) / len(vec_list)
                #corr = cosine(target_vec,averaged_vec)
                corr,_ = pearsonr(target_vec,averaged_vec)
                dissi_vals.append(1-corr)
                                  
    return dissi_vals

def match_tokens(words,encodings):
    
    word_list = [w.lower() for w in words]
    matches = {}
    input_ids = [int(val) for val in encodings['input_ids'][0]]
    
    multi_token = False
    accum = ''
    init_id = 0
    w_id = 0
    
    for i in range(len(input_ids)):
        
        
        word = word_list[w_id]
        token_id = input_ids[i]
       
        # reconstruct the token
            
        token = re.search("[\w\d\']+",tokenizer.convert_ids_to_tokens(token_id)).group()
       
        if token == word and multi_token == False:
            # update matches
            matches[token] = [i]
            # update initial index
            init_id += 1
            w_id += 1
        else: 
            multi_token = True
            accum += token
            # check if the new token matches a word
            if accum == word:
                matches[accum] = list(range(init_id,(i+1)))
                # reset the flags
                multi_token = False
                accum = ''
                # update init_id
                init_id = i+1
                # only when a word is matched, will we go on to check the next word
                w_id += 1
            else:
                pass
            
    return matches
            
def bert_vals(sent_list):
    
    def retrieve_vec(token_range, vectors):
        if len(token_range) == 1:
            word_onset = token_range[0]
            target_vec = vectors[word_onset].tolist()
        else:
            word_onset = token_range[0]
            word_coda = token_range[-1]
            # if a word has more than one tokens, we apply pooling
            target_vec = torch.mean(vectors[word_onset:word_coda],dim=0).tolist()
            
        return word_onset, target_vec
    
    def cosine_sim(target_vec, vec_list):
        
        averaged_vec = torch.mean(vec_list,dim=0).tolist()
        corr= cosine(target_vec,averaged_vec)
        
        return corr
    
    def pearson_corr(target_vec, vec_list):
        
        averaged_vec = torch.mean(vec_list,dim=0).tolist()
        corr,_ = pearsonr(target_vec,averaged_vec)
        
        return corr
        
    dissi_vals = []
    for i in range(len(sent_list)):
        
        token_list = [t.lower() for t in sent_list[i]]
        sent = ' '.join(token_list)
        encodings = tokenizer(sent,add_special_tokens=False, return_tensors='pt')
        matches = match_tokens(token_list, encodings)
        vectors = encoder(**encodings).last_hidden_state[0]

        if i == 0:
            for n in range(len(token_list)):
                if n == 0:
                    dissi_vals.append(0)
                else:
                    # retrieve the idx or idx range
                    token_range = matches[token_list[n]]
                    word_onset, target_vec = retrieve_vec(token_range,vectors)
                    # retrieve the vectors of the context
                    vec_list = vectors[:word_onset]
                    # compute correlation value
                    #corr = cosine_sim(target_vec,vec_list)
                    corr = pearson_corr(target_vec,vec_list)
                    dissi_vals.append(1-corr)
                    
        else:
            for n in range(len(token_list)):
                token_range = matches[token_list[n]]
                word_onset, target_vec = retrieve_vec(token_range,vectors)
        
                vec_list = vectors[:word_onset] if n != 0 else encoder(**tokenizer(' '.join(sent_list[i-1]),add_special_tokens=False,return_tensors='pt')).last_hidden_state[0]
                #corr = cosine_sim(target_vec,vec_list)
                corr = pearson_corr(target_vec, vec_list)
                
                dissi_vals.append(1-corr)
                                  
    return dissi_vals
            
def calculate_dissi(data):
                                  
    offsets_sf, sent_list, boundaries, onsets = process_stim(data)
    w2v = word2vec_vals(sent_list)
    bert = bert_vals(sent_list)
                                  
    # align the onsets and values
    return list(zip(onsets,w2v,bert))

def align_with_eeg(data):
    
    expanded_data = []
    time_points = [row[1].to_list() for row in data.iterrows()]
    
    onset_id = 0
    coda_id = 0
    for p in time_points:
        if p[1] == p[2] == 0 and coda_id != 0:
            sub_data = time_points[onset_id+1:coda_id]
            onset_id = coda_id
            expanded_data.append(expand_data(sub_data))
        
        coda_id += 1
        
    return expanded_data


def expand_data(data):
    
    sample_points = [num/128 for num in range(22912)]
    output = [0] * 22912
    
    for p in data:
        closest = min(sample_points, key = lambda x: abs(x-p[0]))
        index = sample_points.index(closest)
        output[index] = (p[1],p[2])
    
    for i in range(1,22912):
        if output[i] == 0:
            output[i] = output[i-1]
        else:
            pass
        
    return output

paths = glob(r'E:/PhD/data/Di_Liberto/Di_Liberto/Natural Speech/Stimuli/Text/*.mat')
# four datasets are removed because they contain words that cannot be processed by the word2vec model
paths.pop(16)
paths.pop(15)
paths.pop(9)
paths.pop(8)

onset_vals = []
for i,p in enumerate(paths):
    start_time = time()
    data = sio.loadmat(p)
    onset_vals.append(calculate_dissi(data))
    end_time = time()
    print('--------------------------------------------------------------------------------------------------')
    print(f'file {i} completed; time: {end_time-start_time}')
    
data = []
for x in onset_vals:
    for s in x:
        data.append(s)
df = pd.DataFrame.from_records(data,columns=['onset','w2v','bert'])

expanded_data = align_with_eeg(df)
#with open(r'E:/PhD/data/Di_Liberto/value_cosine.pkl','wb') as file:
    #pickle.dump(expanded_data, file)
with open(r'E:/PhD/data/Di_Liberto/value_pearson.pkl','wb') as file:
    #pickle.dump(expanded_data, file)

