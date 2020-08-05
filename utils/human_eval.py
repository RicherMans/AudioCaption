import os
import sys
import copy
import pickle

import numpy as np
import pandas as pd
import fire

sys.path.append(os.getcwd())


def coco_score(refs, scorer):
    if scorer.method() == "Bleu":
        scores = np.array([ 0.0 for n in range(4) ])
    else:
        scores = 0
    num_cap_per_audio = len(refs[list(refs.keys())[0]])

    for i in range(num_cap_per_audio):
        if i > 0:
            for key in refs:
                refs[key].insert(0, res[key][0])
        res = {key: [refs[key].pop(),] for key in refs}
        score, _ = scorer.compute_score(refs, res)    
        
        if scorer.method() == "Bleu":
            scores += np.array(score)
        else:
            scores += score
    
    score = scores / num_cap_per_audio
    return score

   
def main(eval_caption_file, output, zh=False):
    df = pd.read_json(eval_caption_file)
    if zh:
        refs = df.groupby("key")["tokens"].apply(list).to_dict()
    else:
        refs = df.groupby("key")["caption"].apply(list).to_dict()

    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    scorer = Bleu(zh=zh)
    bleu_scores = coco_score(copy.deepcopy(refs), scorer)
    scorer = Cider(zh=zh)
    cider_score = coco_score(copy.deepcopy(refs), scorer)
    scorer = Rouge(zh=zh)
    rouge_score = coco_score(copy.deepcopy(refs), scorer)

    if not zh:
        from pycocoevalcap.meteor.meteor import Meteor
        scorer = Meteor()
        meteor_score = coco_score(copy.deepcopy(refs), scorer)

        from pycocoevalcap.spice.spice import Spice
        scorer = Spice()
        spice_score = coco_score(copy.deepcopy(refs), scorer)
    

    with open(output, "w") as f:
        for n in range(4):
            f.write("BLEU-{}: {:6.3f}\n".format(n+1, bleu_scores[n]))
        f.write("CIDEr: {:6.3f}\n".format(cider_score))
        f.write("ROUGE: {:6.3f}\n".format(rouge_score))
        if not zh:
            f.write("Meteor: {:6.3f}\n".format(meteor_score))
            f.write("SPICE: {:6.3f}\n".format(spice_score))


if __name__ == "__main__":
    fire.Fire(main)
