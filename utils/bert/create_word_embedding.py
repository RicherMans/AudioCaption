# -*- coding: utf-8 -*-

import sys
import os

sys.path.append(os.getcwd())

from bert_serving.client import BertClient
from utils.build_vocab import Vocabulary
import numpy as np
from tqdm import tqdm
import pickle
import fire
import torch


def main(vocab_file: str, output: str):
    bc = BertClient(ip='localhost')
    vocabulary = torch.load(vocab_file)
    vocab_size = len(vocabulary)
    bert_embeddings = np.zeros((vocab_size, 768))
    for i in tqdm(range(len(bert_embeddings)), ascii=True):
        bert_embeddings[i] = bc.encode([vocabulary.idx2word[i]])
    # with open(out_file, 'wb') as f:
        # pickle.dump(bert_embeddings, f)
    np.save(output, bert_embeddings)


if __name__ == '__main__':
    fire.Fire(main)

    
