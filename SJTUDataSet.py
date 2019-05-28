import kaldi_io
import torch
import torch.utils.data as data
from build_vocab import Vocabulary
import pandas as pd
from nltk.parse.corenlp import CoreNLPParser
import numpy as np
import os
import pickle 


class SJTUDataLoader(data.Dataset):

    def __init__(self, kaldi_string, caption_json_path,
                 vocab_path, sent_embedding_path, transform=None
                 ):
        """Dataloader for the SJTU Audiocaptioning dataset

        Args:
            kaldi_string (string): Kaldi command to load the data (e.g., copy-feats ark:- ark:- |)
            caption_json_path (string): Path to the captioning
            vocab_path (string): Path to the vocabulary (preprocessed by build_vocab)
            transform (function, optional): Defaults to None. Transformation onto the data (function)
        """
        self.dataset = {k: v for k, v in kaldi_io.read_mat_ark(kaldi_string)}
        self.transform = transform
        self.captions = pd.read_json(
            caption_json_path)
        idx_to_fname = self.captions['filename'].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0])
        self.indextodataid = list(
            idx_to_fname[idx_to_fname.isin(list(self.dataset.keys()))]) 
        self.vocab = torch.load(vocab_path)
        if sent_embedding_path:
            with open(sent_embedding_path, 'rb') as f:
                self.sent_embeddings = pickle.load(f)


    def __getitem__(self, index: int):
        dataid = self.indextodataid[index]
        # caption = self.captions.iloc[[index]]['caption'].to_string()
        dataset = self.dataset
        vocab = self.vocab
        feature = dataset[dataid]
        tokens = self.captions.iloc[[index]]['tokens'].tolist()[0]
        caption = [vocab('<start>')] + [vocab(token)
                                        for token in tokens] + [vocab('<end>')]
        sent_embedding = self.sent_embeddings[index]
        if self.transform:
            feature = self.transform(feature)
        return torch.tensor(feature), torch.tensor(caption), torch.tensor(sent_embedding)

    def __len__(self):
        return len(self.indextodataid)


def collate_fn(data_batches):
    data_batches.sort(key=lambda x: len(x[1]), reverse=True)

    def merge_seq(dataseq, dim=0):
        lengths = [seq.shape for seq in dataseq]
        # Assuming duration is given in the first dimension of each sequence
        maxlengths = tuple(np.max(lengths, axis=dim))
        # For the case that the lengths are 2 dimensional
        lengths = np.array(lengths)[:, dim]
        padded = torch.zeros((len(dataseq),) + maxlengths)
        for i, seq in enumerate(dataseq):
            end = lengths[i]
            padded[i, :end] = seq[:end]
        return padded, lengths
    features, captions, sent_embeddings = zip(*data_batches)
    features_seq, feature_lengths = merge_seq(features)
    targets_seq, target_lengths = merge_seq(captions)
    sent_embedding_seq, sent_embedding_lengths = merge_seq(sent_embeddings)

    return features_seq, targets_seq, sent_embedding_seq, target_lengths


def create_dataloader(
        kaldi_string, caption_json_path, vocab_path, sent_embedding_path, transform=None,
        shuffle=True, batch_size: int = 16, num_workers=1,**kwargs
        ):
    dataset = SJTUDataLoader(
        kaldi_string=kaldi_string,
        caption_json_path=caption_json_path,
        sent_embedding_path=sent_embedding_path,
        vocab_path=vocab_path,
        transform=transform)

    return data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=shuffle, collate_fn=collate_fn, **kwargs)

def create_dataloader_train_cv(
        kaldi_string, caption_json_path, vocab_path, sent_embeddings_path, transform=None,
        shuffle=True, batch_size: int = 16, num_workers=1, percent = 90,
        ):
    dataset = SJTUDataLoader(
        kaldi_string=kaldi_string,
        caption_json_path=caption_json_path,
        sent_embedding_path=sent_embedding_path,
        vocab_path=vocab_path,
        transform=transform)
    all_indices = torch.arange(len(dataset))
    num_train_indices = int(len(all_indices) * percent / 100)
    train_indices = all_indices[:num_train_indices]
    cv_indices = all_indices[num_train_indices:]
    trainsampler=data.SubsetRandomSampler(train_indices)
    # Do not shuffle
    cvsampler = SubsetSampler(cv_indices)

    return data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        collate_fn=collate_fn, sampler=trainsampler),data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        collate_fn=collate_fn, sampler=cvsampler)

class SubsetSampler(data.Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'vocab',
        default='data/vocab_hospital.th',
        type=str,
        nargs="?")
    args = parser.parse_args()
    dataset = SJTUDataLoader(
        "data/logmelspect/64dim/hospital.ark",
        "data/filelabel_merged/hospital_tokenized.json",
        vocab_path=args.vocab)
    # for feat, target in dataset:
    # print(feat.shape, target.shape)
    dsetloader = data.DataLoader(
        dataset,
        batch_size=16,
        collate_fn=collate_fn,
        num_workers=4,
        shuffle=True)
    agg = 0
    for feat, target, lengths in dsetloader:
        agg += len(feat)
        print(feat.shape, target.shape, lengths)
    print("Overall seen {} feats (of {})".format(agg, len(dataset)))
    traindataloader, cvdataloader = create_dataloader_train_cv("copy-feats ark:data/logmelspect/64dim/hospital.ark ark:- |",
        "data/filelabel_merged/hospital_tokenized.json",vocab_path=args.vocab)
    for f, t, l in cvdataloader:
        print(f.shape, t. shape, l)


