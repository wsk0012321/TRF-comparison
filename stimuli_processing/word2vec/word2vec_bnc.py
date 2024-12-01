import gensim
from glob import glob
from bs4 import BeautifulSoup
from tqdm import tqdm

def extract_text(path_list):
    
    token_list = []
    for path in path_list: 
        file = open(path, encoding='utf-8').read()
        soup = BeautifulSoup(file, 'xml')
        for sent in soup.find_all('s'):
            tokens_in_sent = [w.text.strip().upper()
                             for w in sent.find_all('w')
                             ]    
            token_list.append(tokens_in_sent)
    
    return token_list

path_list = glob(r'E:/PhD/CORPUS/BNC/Texts/*/*/*.xml')
token_list = extract_text(tqdm(path_list))
print('Extraction completed. Processing modelling...')
model = Word2Vec(sentences=token_list, vector_size=400,window=11, min_count=1,sg=0, workers=4)
print('Finished.')
model.save(r'E:/PhD/data/Di_Liberto/word2vec_bnc_400d.model')

