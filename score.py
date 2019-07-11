#!/usr/bin/env python3
import torch
from build_vocab import Vocabulary
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
import sys
from models import *
import os
import fire
import kaldi_io
import numpy as np
from contextlib import contextmanager
from train import parsecopyfeats


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


def score(data_path: str, encoder_path: str, vocab_path: str,
          captions_file: str, sample_length: int = 30, N=4, smoothing='method1', output: str=None):
    dump = torch.load(encoder_path, map_location=lambda storage, loc: storage)
    reference_df = pd.read_json(captions_file)
    reference_df['filename'] = reference_df['filename'].apply(
        lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    reference_grouped_df = reference_df.groupby(
        ['filename'])['tokens'].apply(list).to_dict()
    encodermodel = dump['encodermodel']
    decodermodel = dump['decodermodel']
    # Some scaler (sklearn standardscaler)
    scaler = dump['scaler']
    # Also load previous training config
    config_parameters = dump['config']

    vocab = torch.load(vocab_path)
    # load images from previous
    encodermodel = encodermodel.to(DEVICE).eval()
    decodermodel = decodermodel.to(DEVICE).eval()
    smoother = SmoothingFunction()
    smoothing_fun = getattr(smoother, smoothing)
    kaldi_string = parsecopyfeats(
        data_path, **config_parameters['feature_args'])
    bleu_score = []
    human_bleu_score = []
    bleu_weights = [1./N]*N
    with stdout_or_file(output) as writer:
        with torch.no_grad():
            for k, features in kaldi_io.read_mat_ark(kaldi_string):
                k = int(k)
                if k not in reference_grouped_df:
                    continue
                features = scaler.transform(features)
                # Add single batch dimension
                features = torch.from_numpy(features).to(DEVICE).unsqueeze(0)
                # Generate an caption embedding
                encoded_feature, hiddens = encodermodel(features)
                sampled_ids = decodermodel.sample(
                    encoded_feature, states=hiddens, maxlength=sample_length)
                # (1, max_seq_length) -> (max_seq_length)
                sampled_ids = sampled_ids[0].cpu().numpy()

                # Convert word_ids to words
                candidate = []
                for word_id in sampled_ids:
                    word = vocab.idx2word[word_id]
                    # Dont add start, end tokens
                    if word == '<end>':
                        break
                    elif word == '<start>':
                        continue
                    candidate.append(word)
                reference_sent = reference_grouped_df[k]
                #human_avg_score = []
                #bleu_avg_score = []
                human_scores = []
                system_scores = []
                for turn in range(len(reference_sent)):
                    human_cand = reference_sent[turn]
                    human_ref = [x for i, x in enumerate(
                        reference_sent) if i != turn]
                    #human_avg_score.append(
                    human_scores.append(
                        sentence_bleu(
                            human_ref, human_cand,
                            smoothing_function=smoothing_fun,
                            weights=bleu_weights))
                    #bleu_avg_score.append(
                    system_scores.append(
                        sentence_bleu(
                            human_ref,
                            candidate,
                            smoothing_function=smoothing_fun,
                            weights=bleu_weights))


                #human_bleu = sum(human_scores)/len(human_scores)
                human_bleu = max(human_scores)
                #bleu_score_all_ref = sum(system_scores)/len(system_scores)
                bleu_score_all_ref = max(system_scores)

                human_bleu_score.append(human_bleu)
                bleu_score.append(bleu_score_all_ref)

            writer.write("BLEU-{} Scores\n".format(N))
            writer.write("System {:10.3f}\n".format(np.mean(bleu_score)))
            writer.write("Human {:10.3f}\n".format(np.mean(human_bleu_score)))


if __name__ == "__main__":
    fire.Fire(score)
