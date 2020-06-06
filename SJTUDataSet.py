import os
import pandas as pd
import numpy as np
import torch
import utils.kaldi_io as kaldi_io
from utils.build_vocab import Vocabulary


class SJTUDataset(torch.utils.data.Dataset):

    def __init__(self, kaldi_stream, caption_df,
                 vocabulary, transform=None):
        """Dataset for the SJTU Audiocaptioning dataset

        Args:
            kaldi_stream (string): Kaldi command to load the data (e.g., copy-feats ark:- ark:- |)
            caption_df (pd.DataFrame): Captioning dataframe
            vocab_file (string): Path to the vocabulary (preprocessed by build_vocab)
            transform (function, optional): Defaults to None. Transformation onto the data (function)
        """
        super(SJTUDataset, self).__init__()
        self._dataset = {k: v for k, v in kaldi_io.read_mat_ark(kaldi_stream)}
        self._transform = transform
        self._caption_df = caption_df
        # idx_to_fname = self._caption_df['filename'].apply(
            # lambda x: os.path.splitext(os.path.basename(x))[0]) # pd.Series, index -> filename
        # self._indextodataid = list(
            # idx_to_fname[idx_to_fname.isin(list(self._dataset.keys()))]) 
        self._vocabulary = vocabulary

    def __getitem__(self, index: int):
        # dataid = self._indextodataid[index]
        dataid = self._caption_df.iloc[index]["key"]
        feature = self._dataset[dataid]
        tokens = self._caption_df.iloc[[index]]['tokens'].tolist()[0]
        caption = [self._vocabulary('<start>')] + \
            [self._vocabulary(token) for token in tokens] + \
            [self._vocabulary('<end>')]

        if self._transform:
            feature = self._transform(feature)

        return torch.as_tensor(feature), torch.as_tensor(caption), dataid

    def __len__(self):
        return len(self._caption_df)

class SJTUDatasetSentence(SJTUDataset):

    def __init__(self, kaldi_stream, caption_df, 
                 sent_embeds, vocabulary, transform=None):
        """Add sentence embeddings
        Args:
            sent_embeds (dict): Dict containing sentence embeddings for training, can be accessed by "<key>_<caption_index>"
        """
        super(SJTUDatasetSentence, self).__init__(
            kaldi_stream, caption_df, vocabulary, transform)
        self._sent_embeds = sent_embeds

    def __getitem__(self, index: int):
        dataid = self._caption_df.iloc[index]["key"]
        caption_idx = self._caption_df.iloc[index]["caption_index"]
        feature = self._dataset[dataid]
        tokens = self._caption_df.iloc[[index]]["tokens"].tolist()[0]
        caption = [self._vocabulary('<start>')] + \
            [self._vocabulary(token) for token in tokens] + \
            [self._vocabulary('<end>')]
        sent_embed = self._sent_embeds["{}_{}".format(dataid, caption_idx)]

        if self._transform:
            feature = self._transform(feature)

        return torch.as_tensor(feature), torch.as_tensor(caption), \
                torch.as_tensor(sent_embed), dataid



class SJTUDatasetEval(torch.utils.data.Dataset):
    
    def __init__(self, kaldi_stream, kaldi_scp, transform=None):
        super(SJTUDatasetEval, self).__init__()
        self._kaldi_scp = kaldi_scp
        self._data_generator = kaldi_io.read_mat_ark(kaldi_stream)
        self._transform = transform

    def __getitem__(self, index):
        key, feature = next(self._data_generator)
        if self._transform:
            feature = self._transform(feature)
        return key, torch.as_tensor(feature)

    def __len__(self):
        with open(self._kaldi_scp, "r") as f:
            length = len(f.readlines())
        return length


def collate_fn(length_idxs: int):

    def collate_wrapper(data_batches):
        # x: [feature, caption]
        # data_batches: [[feat1, cap1], [feat2, cap2], ..., [feat_n, cap_n]]
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
        
        data_out = []
        data_len = []
        for idx, data in enumerate(zip(*data_batches)):
            if isinstance(data[0], torch.Tensor):
                if len(data[0].shape) == 0:
                    data_seq = torch.as_tensor(data)
                elif data[0].size(0) > 1:
                    data_seq, tmp_len = merge_seq(data)
                    if idx in length_idxs:
                        data_len.append(tmp_len)
            else:
                data_seq = data
            data_out.append(data_seq)
        data_out.extend(data_len)

        return data_out

    return collate_wrapper


def create_dataloader(
        kaldi_stream, caption_df, vocabulary, transform=None,
        shuffle=True, batch_size: int = 16, num_workers=1,**kwargs
        ):
    dataset = SJTUDataset(
        kaldi_stream=kaldi_stream,
        caption_df=caption_df,
        vocabulary=vocabulary,
        transform=transform)

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=shuffle, collate_fn=collate_fn([0, 1]), **kwargs)


def create_dataloader_train_cv(kaldi_stream,
                               caption_df,
                               vocabulary,
                               transform=None,
                               batch_size: int = 16,
                               num_workers=4,
                               percent=90,
                               **kwargs):
    train_df = caption_df.sample(frac=percent / 100., random_state=0)
    cv_df = caption_df[~caption_df.index.isin(train_df.index)]

    return create_dataloader(kaldi_stream=kaldi_stream,
                             caption_df=caption_df,
                             vocabulary=vocabulary,
                             transform=transform,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers,
                             **kwargs), create_dataloader(
                                 kaldi_stream=kaldi_stream,
                                 caption_df=caption_df,
                                 vocabulary=vocabulary,
                                 transform=transform,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 **kwargs)

class SubsetSampler(torch.utils.data.Sampler):
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
        'vocab_file',
        default='data/car/vocab_zh.pth',
        type=str,
        nargs="?")
    args = parser.parse_args()
    caption_df = pd.read_json("data/car/labels/car_ch.json")
    vocabulary = torch.load(args.vocab_file)
    dataset = SJTUDataset(
        "copy-feats scp:data/car/feats.scp ark:- |",
        caption_df,
        vocabulary)
    # for feat, target in dataset:
    # print(feat.shape, target.shape)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        collate_fn=collate_fn,
        num_workers=4,
        shuffle=True)
    agg = 0
    for feat, target, lengths in dataloader:
        agg += len(feat)
        print(feat.shape, target.shape, lengths)
    print("Overall seen {} feats (of {})".format(agg, len(dataset)))
    traindataloader, cvdataloader = create_dataloader_train_cv(
            "copy-feats scp:data/car/feats.scp ark:- |",
            caption_df,
            vocabulary)
    for f, t, l in cvdataloader:
        print(f.shape, t.shape, l)


