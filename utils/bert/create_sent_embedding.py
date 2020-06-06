import torch
from tensorboardX import SummaryWriter
import pandas as pd
import numpy as np
import os
from bert_serving.client import BertClient
from tqdm import tqdm
import pickle
import fire


def main(caption_file: str, output: str, train: bool=True):
    df = pd.read_json(caption_file)
    bc = BertClient()
    embeddings = {}

    if train:
        with tqdm(total=df.shape[0]) as pbar:
            for idx, row in df.iterrows():
                caption = row["caption"]
                key = row["key"]
                caption_index = row["caption_index"]
                embeddings["{}_{}".format(key, caption_index)] = bc.encode([caption]).reshape(-1)
                pbar.update()

    else:
        dump = {}

        for i in tqdm(range(len(df))):
            sub_df = df.iloc[i]
            key = sub_df["key"]
            caption = sub_df.caption
            value = bc.encode([caption])

            if key not in embeddings.keys():
                embeddings[key] = [value]
            else:
                embeddings[key].append(value)
            
        for key in embeddings:
            dump[key] = np.concatenate(embeddings[key])

        embeddings = dump

    with open(output, 'wb') as f:
        pickle.dump(embeddings, f)


if __name__ == '__main__':
    fire.Fire(main)

