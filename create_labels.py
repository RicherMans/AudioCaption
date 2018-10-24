# -*- coding: utf-8 -*-
import fire
import pandas as pd
from nltk.parse.corenlp import CoreNLPParser
from tqdm import tqdm
import io
import sys
from zhon.hanzi import punctuation
import re
# Fix for chinese characters ...
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')


def process(input_labels_csv: str, output_file: str,
            hostname="http://localhost:9000", character_level: bool = False):
    captions = pd.read_csv(input_labels_csv, sep='\t', encoding='utf-8')
    parser = CoreNLPParser(hostname)
    captions = captions[captions.caption.notnull()]
    captions['tokens'] = None
    for idx, row in tqdm(captions.iterrows(), total=len(captions)):
        caption = row['caption']
        # Remove punctuation
        caption = re.sub("[{}]".format(punctuation),"",caption)
        if character_level:
            captions.at[idx, 'tokens'] = list(caption)
        else:
            captions.at[idx, 'tokens'] = list(parser.tokenize(caption))
    captions.to_json(output_file)


if __name__ == "__main__":
    fire.Fire(process)
