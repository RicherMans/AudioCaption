# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class BaseDecoder(nn.Module):
    """
    Take word/audio embeddings and output the next word probs
    Base decoder, cannot be called directly
    All decoders should inherit from this class
    """

    def __init__(self, input_size, vocab_size):
        super(BaseDecoder, self).__init__()
        self.input_size = input_size
        self.vocab_size = vocab_size

    def forward(self, x):
        raise NotImplementedError


class RNNDecoder(BaseDecoder):

    def __init__(self, input_size, vocab_size, **kwargs):
        super(RNNDecoder, self).__init__(input_size, vocab_size)
        hidden_size = kwargs.get('hidden_size', 256)
        num_layers = kwargs.get('num_layers', 1)
        bidirectional = kwargs.get('bidirectional', False)
        rnn_type = kwargs.get('rnn_type', "GRU")
        self.model = getattr(nn, rnn_type)(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional)
        self.classifier = nn.Linear(
            hidden_size * (bidirectional + 1), vocab_size)
        nn.init.kaiming_uniform_(self.classifier.weight)
    
    def forward(self, *input, **kwargs):
        """
        RNN-style decoder must implement `forward` like this:
            accept a word embedding input and last time hidden state, return the word
            logits output and hidden state of this timestep
        the return dict must contain at least `logits` and `states`
        """
        if len(input) == 1:
            x = input # x: input word embedding/feature at timestep t
            states = None
        elif len(input) == 2:
            x, states = input
        else:
            raise Exception("unknown input type for rnn decoder")

        out, states = self.model(x, states)
        # out: [N, 1, hs], states: [num_layers * num_directionals, N, hs]

        output = {
            "states": states,
            "logits": self.classifier(out)
        }

        # if "seq_output" in kwargs and kwargs["seq_output"]:
            # if isinstance(out, nn.utils.rnn.PackedSequence):
                # padded_out, lens = nn.utils.rnn.pad_packed_sequence(
                    # out, batch_first=True)
                # N, T, _ = padded_out.shape
                # idxs = torch.arange(T, device="cpu").repeat(N).view(N, T)
                # mask = (idxs < lens.view(-1, 1)).to(padded_out.device)
                # # mask: [N, T]
                # seq_outputs = padded_out * mask.unsqueeze(-1)
                # seq_outputs = seq_outputs.sum(1)
                # seq_outputs = seq_outputs / lens.unsqueeze(1).to(padded_out.device)
                # # seq_outputs: [N, E]
                # output["seq_outputs"] = seq_outputs
            # else:
                # output["hidden"] = out

        return output

    def init_hidden(self, bs):
        bidirectional = self.model.bidirectional
        num_layers = self.model.num_layers
        hidden_size = self.model.hidden_size
        return torch.zeros((bidirectional + 1) * num_layers, bs, hidden_size)


class RNNLuongAttnDecoder(RNNDecoder):

    def __init__(self, 
                 input_size,
                 attn_hidden_size, 
                 vocab_size,
                 **kwargs):
        super(RNNLuongAttnDecoder, self).__init__(input_size, vocab_size, **kwargs)
        self.hc2attn_h = nn.Linear(input_size + self.model.hidden_size, attn_hidden_size)
        self.classifier = nn.Linear(attn_hidden_size, vocab_size)
        nn.init.kaiming_uniform_(self.hc2attn_h.weight)
        nn.init.kaiming_uniform_(self.classifier.weight)

    def forward(self, *rnn_input, **attn_args): 
        x, h = rnn_input
        attn = attn_args["attn"]
        h_enc = attn_args["h_enc"]
        src_lens = attn_args["src_lens"]

        x = x.unsqueeze(1)
        out, h = self.model(x, h)
        c, attn_weight = attn(h.squeeze(0), h_enc, src_lens)
        attn_h = torch.tanh(self.hc2attn_h(torch.cat((h.squeeze(0), c), dim=-1)))
        logits = self.classifier(attn_h)

        return {"states": h, "logits": logits, "weights": attn_weight}


class RNNBahdanauAttnDecoder(RNNDecoder):

    def __init__(self, 
                 input_size, 
                 vocab_size,
                 **kwargs):
        super(RNNBahdanauAttnDecoder, self).__init__(input_size, vocab_size, **kwargs)
        self.classifier = nn.Linear(self.model.hidden_size, vocab_size)
        nn.init.kaiming_uniform_(self.classifier.weight)
    
    def forward(self, *rnn_input, **attn_args):
        x, h = rnn_input
        attn = attn_args["attn"]
        h_enc = attn_args["h_enc"]
        src_lens = attn_args["src_lens"]

        c, attn_weight = attn(h.squeeze(0), h_enc, src_lens)
        rnn_input = torch.cat((x, c), dim=-1).unsqueeze(1)
        out, h = self.model(rnn_input, h)
        logits = self.classifier(out.squeeze(1))

        # print(logits.shape)
        return {"states": h, "logits": logits, "weights": attn_weight}

# class GRUDecoder(RNNDecoder):

    # def __init__(self, embed_size, vocab_size, **kwargs):
        # super(GRUDecoder, self).__init__(embed_size, vocab_size)
        # hidden_size = kwargs.get('hidden_size', 256)
        # num_layers = kwargs.get('num_layers', 1)
        # bidirectional = kwargs.get('bidirectional', False)
        # self.model = nn.GRU(
            # input_size=embed_size,
            # hidden_size=hidden_size,
            # num_layers=num_layers,
            # batch_first=True,
            # bidirectional=bidirectional)
        # self.classifier = nn.Linear(
            # hidden_size * (bidirectional + 1), vocab_size)
        # nn.init.kaiming_uniform_(self.classifier.weight)


# class LSTMDecoder(RNNDecoder):

    # def __init__(self, embed_size, vocab_size, **kwargs):
        # super(LSTMDecoder, self).__init__(embed_size, vocab_size)
        # hidden_size = kwargs.get('hidden_size', 256)
        # num_layers = kwargs.get('num_layers', 1)
        # bidirectional = kwargs.get('bidirectional', False)
        # self.model = nn.LSTM(
            # input_size=embed_size,
            # hidden_size=hidden_size,
            # num_layers=num_layers,
            # batch_first=True,
            # bidirectional=bidirectional)
        # self.classifier = nn.Linear(
            # hidden_size * (bidirectional + 1), vocab_size)
        # nn.init.kaiming_uniform_(self.classifier.weight)



