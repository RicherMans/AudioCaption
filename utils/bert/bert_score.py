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
from tqdm import tqdm
from train import parsecopyfeats
import pickle


def log_cosine_similarity(vec1, vec2):
    s = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return -np.log2(1 - s)


# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def bert_score(data_path: str, model_path: str, vocab_path: str, bert_sent_embed_label: str,
          captions_file: str, sample_length: int = 30, mode: str = 'max'):
    dump = torch.load(model_path, map_location=lambda storage, loc: storage)
    reference_df = pd.read_json(captions_file)
    reference_df['filename'] = reference_df['filename'].apply(
        lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    reference_grouped_df = reference_df.groupby(
        ['num'])['tokens'].apply(list).to_dict()
    # model = dump["model"]
    encodermodel = dump["encodermodel"]
    decodermodel = dump["decodermodel"]
    # Some scaler (sklearn standardscaler)
    scaler = dump['scaler']
    # Also load previous training config
    config_parameters = dump['config']

    vocabulary = torch.load(vocab_path)
    # load images from previous
    # model = model.to(DEVICE).eval()
    encodermodel = encodermodel.to(DEVICE).eval()
    decodermodel = decodermodel.to(DEVICE).eval()

    kaldi_string = parsecopyfeats(
        data_path, **config_parameters['feature_args'])
    bert_score = []
    human_bert_score = []

    with open(bert_sent_embed_label, 'rb') as f:
        reference_embeddings = pickle.load(f)

    # cos = nn.CosineSimilarity(dim=1)
    bc = BertClient()

    assert mode in ('mean', 'max')

    with torch.no_grad():
        for k, features in tqdm(kaldi_io.read_mat_ark(kaldi_string)):

            if int(k) not in reference_grouped_df:
                continue

            features = scaler.transform(features)
            # Add single batch dimension
            features = torch.from_numpy(features).to(DEVICE).unsqueeze(0)
            # Generate an caption embedding
            # sampled_ids = model.sample(features, max_length=sample_length)
            encoded_features, encoder_state = encodermodel(features)
            sampled_ids, _ = decodermodel.sample_greedy(
                encoded_features, states=None, maxlength=sample_length, return_probs=True)
            sampled_ids = sampled_ids[0].cpu().numpy()

            # Convert word_ids to words
            candidate = []
            for word_id in sampled_ids:
                word = vocabulary.idx2word[word_id]
                # Dont add start, end tokens
                if word == "<end>":
                    break
                elif word == "<start>":
                    continue
                candidate.append(word)

            candidate = "".join(candidate)
            candidate_embed = bc.encode([candidate])

            reference_embedding = reference_embeddings[k]
            
            human_scores = []
            system_scores = []
            for turn in range(len(reference_embedding)):
                human_cand_emb = reference_embedding[turn]
                for i in range(len(reference_embedding)):
                    if i != turn:
                        human_scores.append(
                            log_cosine_similarity(human_cand_emb.reshape(-1), reference_embedding[i].reshape(-1))
                        )
                        system_scores.append(
                            log_cosine_similarity(human_cand_emb.reshape(-1), candidate_embed.reshape(-1))
                        )
            
            if mode == 'max':
                human_bert = max(human_scores)
                bert_score_all_ref = max(system_scores)
            else:
                human_bert = np.mean(human_scores)
                bert_score_all_ref = np.mean(system_scores)

            human_bert_score.append(human_bert)
            bert_score.append(bert_score_all_ref)

        # writer.write("System {:10.3f}\n".format(np.mean(bert_score)))
        # writer.write("Human {:10.3f}\n".format(np.mean(human_bert_score)))
        model_score = np.mean(bert_score)
        label_score = np.mean(human_bert_score)
        print("System: {:10.3f}\nHuman {:10.3f}".format(model_score, label_score))
        return model_score, label_score


if __name__ == "__main__":
    fire.Fire(bert_score)

