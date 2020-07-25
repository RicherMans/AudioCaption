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

   
def bleu(refs):
    scorer = Bleu(n=4)
    all_scores = np.array([ 0.0 for n in range(4) ])
    num_cap_per_audio = len(refs[list(refs.keys())[0]])

    # keys = np.random.choice(list(refs.keys()), 15, replace=False)
    # for key in keys:
        # np.random.shuffle(refs[key])

    for i in range(num_cap_per_audio):
        if i > 0:
            for key in refs.keys():
                refs[key].insert(0, hypo[key][0])                                                                                                                                        
        hypo = {key: [refs[key].pop(),] for key in refs.keys()}
        score, scores = scorer.compute_score(refs, hypo)

        score = np.array(score)
        all_scores += score

    all_scores = all_scores / num_cap_per_audio
    return all_scores                 

def cider(gts):
    # scorer += (hypo[0], ref1)    
    total_score = 0

    num_cap_per_audio = len(gts[list(gts.keys())[0]])

    for i in range(num_cap_per_audio):
        if i > 0:
            for key in gts:
                gts[key].insert(0, res[key][0])
        res = {key: [gts[key].pop(),] for key in gts}
        scorer = Cider()    
        score, scores = scorer.compute_score(gts, res)    
        
        total_score += score
    
    score = total_score / num_cap_per_audio
    return score
   
def rouge(gts):
    total_score = 0
    num_cap_per_audio = len(gts[list(gts.keys())[0]])

    for i in range(num_cap_per_audio):
        if i > 0:
            for key in gts:
                gts[key].insert(0, res[key][0])
        res = {key: [gts[key].pop(),] for key in gts}
        scorer = Rouge()    
        score, scores = scorer.compute_score(gts, res)    
        
        total_score += score
    
    score = total_score / num_cap_per_audio
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
