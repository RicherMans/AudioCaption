#!/usr/bin/env python3
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.getcwd())
from build_vocab import Vocabulary
from bert_serving.client import BertClient
import pandas as pd
from models import *
import os
import fire
import kaldi_io
import numpy as np
from contextlib import contextmanager
from train import parsecopyfeats
import pickle


# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@contextmanager
def stdout_or_file(buf=None):
    if buf and buf != '-':
        fp = open(buf, 'w', encoding='utf-8')
    else:
        # Workaround for chinese characters
        fp = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
        # fp = sys.stdout
    try:
        yield fp
    finally:
        if fp is not sys.stdout:
            fp.close()


def score(data_path: str, encoder_path: str, vocab_path: str, bert_sent_embed_label: str,
          captions_file: str, sample_length: int = 50, output: str=None, mode: str = 'max'):
    dump = torch.load(encoder_path, map_location=lambda storage, loc: storage)
    reference_df = pd.read_json(captions_file)
    reference_df['filename'] = reference_df['filename'].apply(
        lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    reference_grouped_df = reference_df.groupby(
        ['num'])['tokens'].apply(list).to_dict()
    encodermodel = dump['encodermodel']
    decodermodel = dump['decodermodel']
    # Some scaler (sklearn standardscaler)
    scaler = dump['scaler']
    # Also load previous training config
    config_parameters = dump['config']

    vocabulary = torch.load(vocab_path)
    # load images from previous
    encodermodel = encodermodel.to(DEVICE).eval()
    decodermodel = decodermodel.to(DEVICE).eval()

    kaldi_string = parsecopyfeats(
        data_path, **config_parameters['feature_args'])
    bert_score = []
    human_bert_score = []

    with open(bert_sent_embed_label, 'rb') as f:
        reference_embeddings = pickle.load(f)

    cos = nn.CosineSimilarity(dim=1)
    bc = BertClient()

    assert mode in ('mean', 'max')

    with stdout_or_file(output) as writer:
        with torch.no_grad():
            for k, features in kaldi_io.read_mat_ark(kaldi_string):

                if int(k) not in reference_grouped_df:
                    continue

                features = scaler.transform(features)
                # Add single batch dimension
                features = torch.from_numpy(features).to(DEVICE).unsqueeze(0)
                # Generate an caption embedding
                encoded_feature, encoder_state = encodermodel(features)

                sampled_ids = decodermodel.sample(
                    encoded_feature, states=encoder_state, maxlength=sample_length)
                sampled_ids = sampled_ids[0].cpu().numpy()

                # Convert word_ids to words
                candidate = []
                for word_id in sampled_ids:
                    word = vocabulary.idx2word[word_id]
                    # Dont add start, end tokens
                    if word == '<end>':
                        break
                    elif word == '<start>':
                        continue
                    candidate.append(word)

                candidate = ''.join(candidate)
                candidate_embed = torch.from_numpy(bc.encode([candidate]))

                reference_embedding = reference_embeddings[k]
                reference_embedding = torch.tensor(reference_embedding)
                
                human_scores = []
                system_scores = []
                for turn in range(len(reference_embedding)):
                    human_cand_emb = reference_embedding[turn]
                    for i in range(len(reference_embedding)):
                        if i != turn:
                            human_scores.append(
                                cos(human_cand_emb, reference_embedding[i])
                            )
                            system_scores.append(
                                cos(human_cand_emb, candidate_embed)
                            )
                
                if mode == 'max':
                    human_bert = max(human_scores)
                    bert_score_all_ref = max(system_scores)
                else:
                    human_bert = np.mean(human_scores)
                    bert_score_all_ref = np.mean(system_scores)

                human_bert_score.append(human_bert)
                bert_score.append(bert_score_all_ref)

            writer.write("Bert Scores\n")
            writer.write("System {:10.3f}\n".format(np.mean(bert_score)))
            writer.write("Human {:10.3f}\n".format(np.mean(human_bert_score)))


if __name__ == "__main__":
    fire.Fire(score)
