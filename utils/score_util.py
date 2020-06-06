import os
import sys
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_serving.client import BertClient

from .build_vocab import Vocabulary

bert_client = None

def log_cosine_similarity(vec1, vec2):
    s = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    # return -np.log(1 - s)
    return s

def compute_bleu_score(decode_res,
                       gt,
                       start_idx,
                       end_idx,
                       vocabulary,
                       N=4,
                       smoothing="method1"):
    """
    Args:
        decode_res: decoding results of model, [B, max_length]
        gts: ground truth sentences, [B, max_length], with padding values
    Return:
        score: averaging score of this batch
    """
    scores = []
    weights = [1.0 / N] * N
    smoothing_func = getattr(SmoothingFunction(), smoothing)
    for i in range(gt.shape[0]):
        # prepare hypothesis
        hypothesis = []
        for t, w_t in enumerate(decode_res[i]):
            if w_t == start_idx:
                continue
            elif w_t == end_idx:
                break
            else:
                hypothesis.append(vocabulary.idx2word[w_t])

        # prepare reference
        reference = []
        for w_t in gt[i]:
            if w_t == start_idx:
                continue
            elif w_t == end_idx:
                break
            else:
                reference.append(vocabulary.idx2word[w_t])

        scores.append(
            sentence_bleu(
                [reference],
                hypothesis,
                weights=weights,
                smoothing_function=smoothing_func
            )
        )

    return np.array(scores)


def compute_bert_score(decode_res,
                       gts,
                       start_idx,
                       end_idx,
                       vocabulary,
                       **kwargs):
    """
    Args:
        decode_res: decoding results of model, [B, max_length]
        gts: ground truth sentences, [B, max_length], with padding values
    Return:
        score: averaging score of this batch
    """
    global bert_client

    scores = []

    if bert_client is None:
        bert_client = BertClient()

    for i in range(decode_res.shape[0]):
        # prepare hypothesis
        hypothesis = []
        for w_t in decode_res[i]:
            if w_t == start_idx:
                continue
            elif w_t == end_idx:
                break
            else:
                hypothesis.append(vocabulary.idx2word[w_t])
        hypothesis = "".join(hypothesis)

        reference = []
        for w_t in gts[i]:
            if w_t == start_idx:
                continue
            elif w_t == end_idx:
                break
            else:
                reference.append(vocabulary.idx2word[w_t])
        reference = "".join(reference)
        embeddings = bert_client.encode([reference, hypothesis])
        scores.append(log_cosine_similarity(embeddings[0], embeddings[1]))
    
    return np.array(scores)

