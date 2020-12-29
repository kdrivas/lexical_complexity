from pathlib import Path
import pandas as pd
import pandarallel
import numpy as np
import torch

import spacy

def get_meta(sentence, ref, nlp, option):
    try:
        temp = []
        doc = nlp(sentence)
        for token in doc:
            if option == 'pos':
                temp.append(token.pos_)
            else:
                temp.append(token.text)
        return temp        
    except:
        return []

class LexDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, positions):
        self.encodings = encodings
        self.labels = labels
        self.positions = positions

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['target_positions'] = torch.tensor(self.positions[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def read_lexical_corpus(split_dir, nlp=None, return_complete_sent=False, window_size=3):
    if 'tsv' in split_dir:
        data = pd.read_csv(split_dir, sep='\t')
    elif 'xlsx' in split_dir:
        data = pd.read_excel(split_dir)
        data.rename(columns={'subcorpus': 'corpus'}, inplace=True)
    data.token.fillna('null', inplace=True)
    
    data['pos_label'] = data.apply(lambda x: get_meta(x.sentence, x.token, nlp, 'pos'), axis=1)
    data['sentence_pre'] = data.apply(lambda x: get_meta(x.sentence, x.token, nlp, 'text'), axis=1)
    
    texts = []
    pos_tags = []
    labels = []
    sentence_raw = []
    target_words = []
    corpus = []
    positions = []
    
    for ix, row in data.iterrows():
        try:
            position = row.sentence_pre.index(row.token)
        except:
            for ix, w in enumerate(row.sentence_pre):
                if row.token in w:
                    position = ix
                    break
                    
        if return_complete_sent:
            texts.append(row.sentence_pre)
        else:
            sentence = ' '.join(row.sentence_pre[(position-window_size+1):position] + [row.sentence_pre[position]] +  row.sentence_pre[position:(position+window_size-1)])
            texts.append(sentence)
            
        tags = row.pos_label[(position-window_size+1):position] + [row.pos_label[position]] + row.pos_label[position:(position+window_size-1)] 
        pos_tags.append(tags)
        positions.append(position)
        labels.append(row.complexity)
        sentence_raw.append(row.sentence_pre)
        target_words.append(row.token)
        corpus.append(row.corpus)

    return np.array(texts), np.array(corpus), np.array(labels), np.array(sentence_raw), np.array(target_words), np.array(positions), np.array(pos_tags)