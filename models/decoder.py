# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class BaseDecoder(nn.Module):
    """
    Take word/audio embeddings and output the next word probs
    Base decoder, cannot be called directly
    All decoders should inherit from this class
    """

    def __init__(self, embed_size, vocab_size):
        super(BaseDecoder, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size

    def forward(self, x):
        raise NotImplementedError

class RNNDecoder(BaseDecoder):

    def __init__(self, embed_size, vocab_size):
        super(RNNDecoder, self).__init__(embed_size, vocab_size)
    
    def forward(self, *input, **kwargs):
        if len(input) == 1:
            x = input # x: input word embedding/feature at timestep t
            states = None
        elif len(input) == 2:
            x, states = input
        else:
            raise Exception("unknown input type for rnn decoder")

        out, states = self.model(x, states)
        # out: PackedSequence(data: [total_len, hidden_size], batch_sizes: [...])
        #    or [N, 1, hidden_size]

        output = {"states": states}

        if "seq_output" in kwargs and kwargs["seq_output"]:
            if isinstance(out, nn.utils.rnn.PackedSequence):
                padded_out, lens = nn.utils.rnn.pad_packed_sequence(
                    out, batch_first=True)
                N, T, _ = padded_out.shape
                idxs = torch.arange(T, device="cpu").repeat(N).view(N, T)
                mask = (idxs < lens.view(-1, 1)).to(padded_out.device)
                # mask: [N, T]
                seq_outputs = padded_out * mask.unsqueeze(-1)
                seq_outputs = seq_outputs.sum(1)
                seq_outputs = seq_outputs / lens.unsqueeze(1).to(padded_out.device)
                # seq_outputs: [N, E]
                output["seq_outputs"] = seq_outputs
            else:
                output["hidden"] = out

        probs = self.classifier(out.data)
        output["probs"] = probs
        return output


class GRUDecoder(RNNDecoder):

    def __init__(self, embed_size, vocab_size, **kwargs):
        super(GRUDecoder, self).__init__(embed_size, vocab_size)
        hidden_size = kwargs.get('hidden_size', 256)
        num_layers = kwargs.get('num_layers', 1)
        bidirectional = kwargs.get('bidirectional', False)
        self.model = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional)
        self.classifier = nn.Linear(
            hidden_size * (bidirectional + 1), vocab_size)
        nn.init.kaiming_uniform_(self.classifier.weight)


class LSTMDecoder(RNNDecoder):

    def __init__(self, embed_size, vocab_size, **kwargs):
        super(LSTMDecoder, self).__init__(embed_size, vocab_size)
        hidden_size = kwargs.get('hidden_size', 256)
        num_layers = kwargs.get('num_layers', 1)
        bidirectional = kwargs.get('bidirectional', False)
        self.model = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional)
        self.classifier = nn.Linear(
            hidden_size * (bidirectional + 1), vocab_size)
        nn.init.kaiming_uniform_(self.classifier.weight)



