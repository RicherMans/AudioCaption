#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from build_vocab import Vocabulary
from models import *
import fire
import kaldi_io
from train import parsecopyfeats
import tableprint as tp
from contextlib import contextmanager
import sys
import pandas as pd

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


def sample(data_path: str, encoder_path: str,
        vocab_path: str, sample_length: int = 30, output: str = None, ch: bool = True):
    dump = torch.load(encoder_path, map_location=lambda storage, loc: storage)
    encodermodel = dump['encodermodel']
    decodermodel = dump['decodermodel']
    # Some scaler (sklearn standardscaler)
    scaler = dump['scaler']
    # Also load previous training config
    config_parameters = dump['config']

    vocab = torch.load(vocab_path)
    # print(encodermodel)
    # print(decodermodel)
    # load images from previous
    encodermodel = encodermodel.to(DEVICE).eval()
    decodermodel = decodermodel.to(DEVICE).eval()

    kaldi_string = parsecopyfeats(
        data_path, **config_parameters['feature_args'])
    width_length = sample_length * 4


    with stdout_or_file(output) as writer:
        writer.write(
            tp.header(
                ["InputUtterance", "Output Sentence"],
                style='grid', width=width_length))
        writer.write('\n')

        sentences = set()
        for k, features in kaldi_io.read_mat_ark(kaldi_string):

            features = scaler.transform(features)
            # Add single batch dimension
            features = torch.from_numpy(features).to(DEVICE).unsqueeze(0)
            # Generate an caption embedding
            encoded_feature, hiddens = encodermodel(features)
            sampled_ids = decodermodel.sample(encoded_feature,states = hiddens, maxlength=sample_length)
            # (1, max_seq_length) -> (max_seq_length)
            sampled_ids = sampled_ids[0].cpu().numpy()

            # Convert word_ids to words
            sampled_caption = []
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                   break
            if ch:
                sentence = ''.join(sampled_caption)
            else:
                sentence = ' '.join(sampled_caption)

            sentences.add(sentence)

            # Print out the image and the generated caption
            writer.write(tp.row([k, sentence], style='grid', width=width_length))
            writer.write('\n')
            writer.flush()
        writer.write(tp.bottom(2, style='grid', width=width_length))
        writer.write('\n')
        writer.write('Number of unique sentences: ' + str(len(sentences)))



if __name__ == '__main__':
    fire.Fire(sample)
