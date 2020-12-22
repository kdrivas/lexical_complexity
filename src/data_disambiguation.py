from xml.dom import minidom
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd
import numpy as np
import torch

class DisDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def get_key_senses(path):
    keys = {}
    with open(path, 'r') as f:
        for l in f.read().split('\n'):
            if len(l):
                tokens = l.split(' ')
                keys[tokens[0]] = tokens[1]
    
    return keys

def get_sentences(xmldoc, filter_tokens, keys, max_len):
    corpus = []
    sentences = xmldoc.findAll("sentence")

    for sentence in sentences:
        input_s = ''
        output_s = []
        count_valid_sense = 0
        for word in sentence.children:
            if word.name == 'wf':
                input_s += word.text + ' '
                output_s.append('O')
            elif word.name == 'instance':
                input_s += word.text + ' '
                if word.text in filter_tokens:
                    output_s.append(keys[word['id']])
                    count_valid_sense += 1
                else:
                    output_s.append('O')
        corpus.append([input_s.strip(), output_s, count_valid_sense])

    return corpus

def read_disambiguation_corpus(max_len=50):
    
    with open('data/disambiguation/raw/train/SemCor/semcor.data.xml', 'r') as f:
        data_semcor = f.read()

    with open('data/disambiguation/raw/test/ALL/ALL.data.xml', 'r') as f:
        data_senseval = f.read()

    single = pd.read_csv('data/raw/lcp_single_train.tsv', sep='\t')
    filter_tokens = single.token.fillna('null').drop_duplicates().values

    xml_semcor = BeautifulSoup(data_semcor, features="xml")
    xml_senseval = BeautifulSoup(data_senseval, features="xml")

    keys_semcor = get_key_senses('data/disambiguation/raw/train/SemCor/semcor.gold.key.txt')
    keys_senseval = get_key_senses('data/disambiguation/raw/test/ALL/ALL.gold.key.txt')

    corpus_semcor = get_sentences(xml_semcor, filter_tokens, keys_semcor, max_len)
    corpus_senseval = get_sentences(xml_senseval, filter_tokens, keys_senseval, max_len)

    sentences = []
    senses = []
    for element in corpus_semcor:
        if element[2] > 0 and len(element[0].split()) <= max_len:
            sentences.append(element[0])
            senses.append(element[1])

    for element in corpus_senseval:
        if element[2] > 0 and len(element[0].split()) <= max_len:
            sentences.append(element[0])
            senses.append(element[1])
    
    return sentences, senses