import torch
from tensorboardX import SummaryWriter
import pandas as pd
import numpy as np
import os
from bert_serving.client import BertClient
from tqdm import tqdm
import pickle
import fire


def main(captions_file: str,  output: str,  embedding_size: int = 768, train: bool=True):
    df = pd.read_json(captions_file)
    bc = BertClient()

    if train:
        captions = df.caption.values
        bert_sentence_embeddings = np.zeros((len(captions), embedding_size))
        for i in tqdm(range(len(captions))):
            caption = captions[i]
            bert_sentence_embeddings[i] = bc.encode([caption])
        
    else:
        bert_sentence_embeddings = {}

        for i in tqdm(range(len(df))):
            sub_df = df.iloc[i]
            key = sub_df['num']
            caption = sub_df.caption
            value = bc.encode([caption])

            if key not in bert_sentence_embeddings.keys():
                bert_sentence_embeddings[key] = [value]
            else:
                bert_sentence_embeddings[key].append(value)

    with open(output, 'wb') as f:
        pickle.dump(bert_sentence_embeddings, f)


if __name__ == '__main__':
    fire.Fire(main)

