import os
import sys
import copy

import numpy as np
import pandas as pd
import fire

sys.path.append(os.getcwd())

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge

def bleu(gts):    
    scorer = Bleu(n=4)    
    
    all_scores = [ 0 for n in range(4) ]

    num_cap_per_audio = len(gts[list(gts.keys())[0]])

    for i in range(num_cap_per_audio):
        if i > 0:
            for key in gts:
                gts[key].insert(0, res[key][0])
        res = {key: [gts[key].pop(),] for key in gts}
        score, scores = scorer.compute_score(gts, res)    
        
        for n in range(4):
            all_scores[n] += score[n] * len(res)
    
    all_scores = np.array(all_scores) / len(res) / num_cap_per_audio
    # for n in range(4):
        # print('BLEU-{}: {:6.3f}'.format(n+1, all_scores[n]))
    return all_scores
   
def cider(gts):
    scorer = Cider()    
    # scorer += (hypo[0], ref1)    
    all_scores = 0

    num_cap_per_audio = len(gts[list(gts.keys())[0]])

    for i in range(num_cap_per_audio):
        if i > 0:
            for key in gts:
                gts[key].insert(0, res[key][0])
        res = {key: [gts[key].pop(),] for key in gts}
        score, scores = scorer.compute_score(gts, res)    
        
        all_scores += score * len(res)
    
    score = all_scores / len(res) / num_cap_per_audio
    # print('CIDEr: {:6.3f}'.format(all_scores))
    return score
   
def rouge(gts):
    scorer = Rouge()    
    all_scores = 0
    num_cap_per_audio = len(gts[list(gts.keys())[0]])

    for i in range(num_cap_per_audio):
        if i > 0:
            for key in gts:
                gts[key].insert(0, res[key][0])
        res = {key: [gts[key].pop(),] for key in gts}
        score, scores = scorer.compute_score(gts, res)    
        
        all_scores += score * len(res)
    
    score = all_scores / len(res) / num_cap_per_audio
    # print('ROUGE: {:6.3f}'.format(all_scores))
    return score
   
def main(eval_caption_file, output):
    df = pd.read_json(eval_caption_file)
    gts = df.groupby(["key"])["tokens"].apply(list).to_dict()
    bleu_scores = bleu(copy.deepcopy(gts))
    cider_score = cider(copy.deepcopy(gts))
    rouge_score = rouge(copy.deepcopy(gts))

    with open(output, "w") as f:
        for n in range(4):
            f.write("BLEU-{}: {:6.3f}\n".format(n+1, bleu_scores[n]))
        f.write("CIDEr: {:6.3f}\n".format(cider_score))
        f.write("ROUGE: {:6.3f}\n".format(rouge_score))



if __name__ == "__main__":
    fire.Fire(main)
