# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from gensim.models import doc2vec
from tqdm import tqdm
import pickle
import fire


def main(caption_file: str, doc2vec_model_path: str, embedding_size: int, output: str):
    df = pd.read_json(caption_file)

    captions = df.tokens.values

    doc2vec_sentence_embeddings = np.zeros((len(captions), embedding_size))
    model = doc2vec.Doc2Vec.load(doc2vec_model_path)

    for i in tqdm(range(len(captions))):
        caption = captions[i]
        doc2vec_sentence_embeddings[i] = model.infer_vector(caption)

    with open(output, 'wb') as f:
        pickle.dump(doc2vec_sentence_embeddings, f)


if __name__ == '__main__':
    fire.Fire(main)

