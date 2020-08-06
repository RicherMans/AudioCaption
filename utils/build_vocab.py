from tqdm import tqdm
import pandas as pd
import logging
import torch
from collections import Counter
import re
import fire

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json:str, threshold:int, keeppunctuation: bool, host_address:str, character_level:bool=False, zh:bool=True ):
    from nltk.parse.corenlp import CoreNLPParser
    from zhon.hanzi import punctuation
    """Build vocabulary from csv file with a given threshold to drop all counts < threshold

    Args:
        csv (string): Input csv file. Needs to be tab separated and having a column named 'caption'
        
        Modiefied:
        json(string): Input json file. Shoud have a column named 'caption'
        threshold (int): Threshold to drop all words with counts < threshold
        keeppunctuation (bool): Includes or excludes punctuation.

    Returns:
        vocab (Vocab): Object with the processed vocabulary
    """
    #df = pd.read_csv(csv, sep='\t')
    df = pd.read_json(json)
    counter = Counter()
    
    if zh:
        parser = CoreNLPParser(host_address)
        for i in tqdm(range(len(df)), leave=False, ascii=True):
            caption = str(df.loc[i]['caption'])
            # Remove all punctuations
            if not keeppunctuation:
                caption = re.sub("[{}]".format(punctuation),"",caption)
            if character_level:
                tokens = list(caption)
            else:
                tokens = list(parser.tokenize(caption))
            counter.update(tokens)
    else:
        punctuation = ',.()'
        for i in tqdm(range(len(df)), leave=False, ascii=True):
            caption = str(df.loc[i]['caption'])
            # Remove all punctuations
            if not keeppunctuation:
                caption = re.sub("[{}]".format(punctuation),"",caption)
            if character_level:
                tokens = list(caption)
            else:
                tokens = caption.split()
            counter.update(tokens)

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def process(input_json:str, output_vocab:str, threshold:int = 1, keeppunctuation: bool = False, character_level: bool = False, host_address:str = "http://localhost:9000", zh: bool=True):
    logger=logging.Logger("Build Vocab")
    logger.setLevel(logging.INFO)
    vocab = build_vocab(json=input_json, threshold=threshold, keeppunctuation=keeppunctuation, host_address=host_address, character_level = character_level, zh=zh)
    torch.save(vocab, output_vocab)
    logger.info("Total vocabulary size: {}".format(len(vocab)))
    logger.info("Saved vocab to '{}'".format(output_vocab))


if __name__ == '__main__':
    fire.Fire(process)
