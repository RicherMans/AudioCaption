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


class GRUDecoder(BaseDecoder):

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

    def forward(self, *input, **kwargs):
        if len(input) == 1:
            x = input # x: input word embedding/feature at timestep t
            states = None
        elif len(input) == 2:
            x, states = input
        else:
            raise Exception("unknown input type for rnn decoder")

        output_sentence = kwargs.get("output_sentence", False)

        out, states = self.model(x, states)
        # out: PackedSequence(data: [total_len, hidden_size], batch_sizes: [...])
        #    or [N, 1, hidden_size]
        output = {"states": states}

        if output_sentence and isinstance(out, nn.utils.rnn.PackedSequence):
            padded_out, lens = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True)
            sentence_ouputs = padded_out.sum(1)
            lens = lens.reshape(-1, 1).expand(*sentence_ouputs.size()).to(x.device)
            sentence_ouputs = sentence_ouputs / lens.float()
            output["sentences"] = sentence_ouputs

        probs = self.classifier(out.data)
        output["probs"] = probs

        return output


class LSTMDecoder(BaseDecoder):

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

        # if isinstance(out, nn.utils.rnn.PackedSequence):
            # padded_out, lens = nn.utils.rnn.pad_packed_sequence(
                # out, batch_first=True)
            # sentence_ouputs = padded_out.sum(1)
            # lens = lens.reshape(-1, 1).expand(*sentence_ouputs.size()).to(x.device)
            # sentence_ouputs = sentence_ouputs / lens.float()
        probs = self.classifier(out.data)
        return {"states": states, "probs": probs}




"""
class GRUDecoder_backup(BaseDecoder):

    def __init__(self, embed_size, vocab_size, **kwargs):
        super(GRUDecoder_backup, self).__init__(embed_size, vocab_size)
        hidden_size = kwargs.get('hidden_size', 256)
        num_layers = kwargs.get('num_layers', 1)
        bidirectional = kwargs.get('bidirectional', False)
        batch_first = kwargs.get('batch_first', True)
        dropout_p = kwargs.get('dropout', 0.0)
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.dropoutlayer = nn.Dropout(dropout_p)
        self.model = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional)
        self.classifier = nn.Linear(
            hidden_size * (bidirectional + 1), vocab_size)
        nn.init.kaiming_uniform_(self.classifier.weight)
        nn.init.kaiming_uniform_(self.word_embeddings.weight)



    def sample(self, feature, states=None, maxlength=20, **kwargs):
        # feature: [N, enc_emb_dim]
        # states: [num_layers, N, enc_hid_dim]
        sample_method = kwargs.get("sample_method", "greedy")
        temperature = kwargs.get("temperature", 1.0)

        batch_size = feature.size(0)
        sampled_seqs = torch.zeros(batch_size, maxlength).fill_(self.end_idx)
        sampled_logprobs = feature.new_zeros((batch_size, maxlength))
        # seqLogprobs = []

        #  feature = feature.unsqueeze(1)
        #  # feature: [N, 1, enc_emb_dim]
        sampled_seqs[:, 0] = self.start_idx
        for t in range(maxlength - 1):
            if t == 0:
                x_t = feature
            else:
                if t == 1:
                    # input <start>
                    w_t = torch.zeros(batch_size).fill_(self.start_idx).long().to(feature.device)
                x_t = self.word_embeddings(w_t)
            # x_t: [N, emb_dim]
            x_t = x_t.unsqueeze(1)
            # x_t: [N, 1, emb_dim]

            #  inp = torch.cat((x_t, feature), dim=2)
            # inp: [N, 1, emb_dim + enc_emb_dim]

            hiddens, states = self.model(x_t, states) # hiddens: [batch_size, 1, hid_dim]
            # Only a single timestep here, squeeze that dimension
            outputs = self.classifier(hiddens.squeeze(1))
            # outputs: [batch_size, vocab_size]
            logprobs = torch.log_softmax(outputs, dim=1)

            # Sample the next word
            if sample_method == "greedy":
                sampledLogprobs, w_t = torch.max(logprobs.detach(), 1) # sampledLogprobs: [batch_size], w_t: [batch_size]
                w_t = w_t.long().detach()
            else:
                prob_prev = torch.exp(logprobs.detach() / temperature)
                w_t = torch.multinomial(prob_prev, 1)
                # w_t: [batch_size, 1]
                sampledLogprobs = logprobs.gather(1, w_t)
                w_t = w_t.view(-1).long()

            if t >= 1:
                # decide whether to stop
                if t == 1:
                    unfinished = w_t != self.end_idx
                else:
                    unfinished = unfinished * (w_t != self.end_idx)
                #  w_t[unfinished == 0] = self.end_idx
                sampled_seqs[:, t] = w_t
                sampled_seqs[:, t][unfinished == 0] = self.end_idx
                sampled_logprobs[:, t] = sampledLogprobs.view(-1)
                if unfinished.sum() == 0:
                    break

        return sampled_seqs, sampled_logprobs


    def sample_greedy(self, feature, states=None, maxlength=20, **kwargs):
        start_idx = kwargs.get("start_idx", 1)
        return_probs = kwargs.get("return_probs", True)

        sampled_token_ids = []
        # feature: [1, enc_emb_dim]
        feature = feature.unsqueeze(1)
        # feature: [1, 1, enc_emb_dim]
        sampled_probs = []
        # get hidden from encoded features
        hiddens, states = self.model(feature, states)
        sampled_token_ids.append(torch.tensor([start_idx]))
        feature = self.word_embeddings(torch.tensor([start_idx]).to(feature.device)).unsqueeze(1)
        for i in range(1, maxlength):
            hiddens, states = self.model(feature, states)
            # Only a single timestep here, squeeze that dimension
            outputs = self.classifier(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_token_ids.append(predicted.cpu())
            sampled_probs.append(outputs)
            predicted = predicted.detach()
            # Prepare next input
            feature = self.word_embeddings(predicted).unsqueeze(1)

        sampled_token_ids = torch.stack(sampled_token_ids, 1)
        if return_probs:
            sampled_probs = torch.stack(sampled_probs, 1)
            return sampled_token_ids, sampled_probs
        return sampled_token_ids
"""
