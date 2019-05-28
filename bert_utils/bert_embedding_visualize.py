import torch
from tensorboardX import SummaryWriter
import pandas as pd
import numpy as np
from bert_serving.client import BertClient
from tqdm import tqdm
import pickle


df = pd.read_json('data/car/labels/train.json')

captions = df.caption.values

out_path = 'train_bert_sentence_embeddings.npy'

generated = True

if generated == True:
    emb_size = 768
    bert_sentence_embeddings = np.zeros((len(captions), emb_size))
    
    bc = BertClient()

    for i in tqdm(range(len(captions))):
        caption = captions[i]
        bert_sentence_embeddings[i] = bc.encode([caption])
        
    with open(out_path, 'wb') as f:
        pickle.dump(bert_sentence_embeddings, f)
else:
    with open('bert_sentence_embedding.npy', 'rb') as f:
        bert_sentence_embeddings = pickle.load(f)
    
    with SummaryWriter() as writer:
        first_100_embedding = bert_sentence_embeddings[:100]
        writer.add_embedding(first_100_embedding, metadata=captions[:100], tag='first 100 sentences')
        writer.add_embedding(bert_sentence_embeddings, metadata=captions, tag='all sentence embeddings')

