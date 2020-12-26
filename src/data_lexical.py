from pathlib import Path
import pandas as pd
import numpy as np
import torch

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
    
def read_lexical_corpus(split_dir, return_complete_sent=False, window_size=3):
    if 'tsv' in split_dir:
        data = pd.read_csv(split_dir, sep='\t')
    elif 'xlsx' in split_dir:
        data = pd.read_excel(split_dir)
        data.rename(columns={'subcorpus': 'corpus'}, inplace=True)
    data.token.fillna('null', inplace=True)
    
    texts = []
    labels = []
    sentence_raw = []
    target_words = []
    corpus = []
    positions = []
    for ix, row in data.iterrows():
        try:
            if return_complete_sent:
                texts.append(row.sentence)
            else:
                words = row.sentence.split(' ')
                tokens = row.sentence.partition(row.token)
                sentence = ' '.join(tokens[0].split(' ')[-window_size:]) + tokens[1] + ' '.join(tokens[2].split(' ')[:window_size])
                texts.append(sentence)
            positions.append(len(tokens[0].split(' ')[-window_size:]))
            labels.append(row.complexity)
            sentence_raw.append(row.sentence)
            target_words.append(row.token)
            corpus.append(row.corpus)
        except:
            print('sentence:', words)
            print('word:', row.token)
            print()

    return np.array(texts), np.array(corpus), np.array(labels), np.array(sentence_raw), np.array(target_words), np.array(positions)