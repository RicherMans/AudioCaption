import os
import fire
import numpy as np
import pandas as pd


def calc_ngram(words, n=2):
    return zip(*[words[i:] for i in range(n)])

def calc_richness(df, N):
    scores = []
    
    ngram2count = {}
    # ngram_number = 0

    for idx, row in df.iterrows():
        ngrams = set(calc_ngram(row["tokens"], n=N))
        # ngram_number += len(ngrams)
        for gram in ngrams:
            if gram not in ngram2count:
                ngram2count[gram] = 1
            else:
                ngram2count[gram] += 1
    
    ngram2freq = {}
    
    for ngram, count in ngram2count.items():
        ngram2freq[ngram] = ngram2count[ngram] / df.shape[0]
            
    for tokens in df["tokens"].values:
        ngrams = set(calc_ngram(tokens, n=N))
        reciprocal_score = 0
        if len(ngrams) == 0:
            continue
        for ngram in ngrams:
            reciprocal_score +=  np.log(1 / ngram2freq[ngram])
        reciprocal_score /= len(ngrams)

        scores.append(reciprocal_score)
    score = sum(scores) / len(scores)
    # print(score)
    return score


def diversity_evaluate(caption: [str, pd.DataFrame], N: int=4, output=None):
    """
    caption: json file containing <key>-<caption> pairs
    """
    if isinstance(caption, pd.DataFrame):
        df = caption
    else:
        df = pd.read_json(caption)
    weights = [1./N] * N
    score = 0
    for n in range(N):
        score += calc_richness(df, n + 1) * weights[n]
    if output is not None:
        with open(os.path.join(os.path.dirname(caption), output), "w") as f:
            f.write("Diversity: {:6.3f}\n".format(score))
    return score

if __name__ == "__main__":
    fire.Fire(diversity_evaluate)

