# -*- coding: utf-8 -*-

import sys
import os

sys.path.append(os.getcwd())

from bert_serving.client import BertClient
from build_vocab import Vocabulary
import numpy as np
from tqdm import tqdm
import pickle
import fire
import torch


def main(vocab_file: str, out_file: str, embedding_size: int = 768):
    bc = BertClient(ip='localhost')
    vocabulary = torch.load(vocab_file)
    vocab_size = len(vocabulary)
    bert_embeddings = np.zeros((vocab_size, embedding_size))
    for i in tqdm(range(len(bert_embeddings))):
        bert_embeddings[i] = bc.encode([vocabulary.idx2word[i]])
    with open(out_file, 'wb') as f:
        pickle.dump(bert_embeddings, f)
    return


if __name__ == '__main__':
    fire.Fire(main)

    
