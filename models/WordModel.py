# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

import utils.score_util as score_util

class CaptionModel(nn.Module):
    """
    Encoder-decoder captioning model, with RNN-style decoder.
    """

    start_idx = 1
    end_idx = 2
    max_length = 20

    def __init__(self, encoder, decoder, **kwargs):
        super(CaptionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.use_hidden:
            assert self.encoder.network.hidden_size == self.decoder.model.hidden_size, \
                "hidden size not compatible while use hidden!"
            assert self.encoder.network.num_layers == self.decoder.model.num_layers, \
                """number of layers not compatible while use hidden!
                please either set use_hidden as False or use the same number of layers"""

        dropout_p = kwargs.get("dropout", 0.0)
        self.embed_size = decoder.embed_size
        self.vocab_size = decoder.vocab_size
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embed_size)
        self.dropoutlayer = nn.Dropout(dropout_p)
        nn.init.kaiming_uniform_(self.word_embeddings.weight)

    @classmethod
    def set_index(cls, start_idx, end_idx):
        cls.start_idx = start_idx
        cls.end_idx = end_idx

    def load_word_embeddings(self, embeddings, tune=True, **kwargs):
        assert embeddings.shape[0] == self.vocab_size, "vocabulary size mismatch!"
        
        embeddings = torch.as_tensor(embeddings).float()
        self.word_embeddings.weight = nn.Parameter(embeddings)
        for para in self.word_embeddings.parameters():
            para.requires_grad = tune

        if embeddings.shape[1] != self.embed_size:
            assert "projection" in kwargs, "embedding size mismatch!"
            if kwargs["projection"]:
                self.word_embeddings = nn.Sequential(
                    self.word_embeddings,
                    nn.Linear(embeddings.shape[1], self.embed_size)
                )

    def forward(self, *input, **kwargs):
        if len(input) != 4 and len(input) != 2:
            raise Exception("number of input should be either 4 (feats, feat_lens, caps, cap_lens) or 2 (feats, feat_lens)!")

        mode = kwargs.get("mode", "forward")
        assert mode in ("forward", "sample"), "unknown running mode"
        # "forward" means teacher forcing training, "sample" means sampling

        if len(input) == 2 and mode == "forward":
            raise Exception("missing caption labels for training!")

        return getattr(self, "_" + mode)(*input, **kwargs)

    def _forward(self, *input, **kwargs):
        """Decode audio feature vectors and generates captions.
           With audio feature and captions as input, i.e., teacher forcing
        """
        feats, feat_lens, caps, cap_lens = input
        encoded, states = self.encoder(feats, feat_lens)
        # encoded: [N, emb_dim], the embedding output by the encoder

        # prepare input to the decoder: encoder output + label embeddings
        embeds = self.word_embeddings(caps)
        embeds = self.dropoutlayer(embeds)
        # embeds: [N, max_len, emb_dim]
        embeds = torch.cat((encoded.unsqueeze(1), embeds), 1)
        # embeds: [N, max_len + 1, emb_dim]

        # prepare packed input to the decoder for efficient training (remove padded zeros)
        # audio feature and the first (max_len - 1) word embeddings are packed
        packed = nn.utils.rnn.pack_padded_sequence(
            embeds, cap_lens, batch_first=True)
        probs = self.decoder(packed, states)["probs"]
        output = {"probs": probs, "embedding_output": encoded}
        return output

    def _sample(self, *input, **kwargs):
        if len(input) == 4:
            feats, feat_lens, _, _ = input
        else:
            feats, feat_lens = input

        # feats: [N, T, F]
        encoded, states = self.encoder(feats, feat_lens)
        # encoded: [N, emb_dim]
        # states: [num_layers, N, enc_hid_dim]
        output = self.sample_core(encoded, states, **kwargs)
        output["embedding_output"] = encoded

        return output
    
    def sample_core(self, encoded, states, **kwargs):
        # optional keyword arguments
        method = kwargs.get("method", "greedy")
        temp = kwargs.get("temperature", 1.0)
        max_length = kwargs.get("max_length", self.max_length)
        assert method in ("greedy", "sample", "beam"), "unknown sampling method"

        if method == "beam":
            beam_size = kwargs.get("beam_size", 5)
            return self.sample_beam(
                encoded, states, max_length=max_length, beam_size=beam_size)

        N = encoded.size(0)
        seqs = torch.zeros(N, max_length, dtype=torch.long).fill_(self.end_idx)
        probs = torch.zeros(N, max_length, self.vocab_size)
        sampled_logprobs = torch.zeros(N, max_length)

        # start sampling
        for t in range(max_length):
            # prepare input word/audio embedding
            if t == 0:
                e_t = encoded
            else:
                e_t = self.word_embeddings(w_t)
                e_t = self.dropoutlayer(e_t)
            # e_t: [N, emb_dim]
            e_t = e_t.unsqueeze(1)

            # feed to the decoder to get states and probs
            outputs = self.decoder(e_t, states)
            states = outputs["states"]
            # outputs["probs"]: [N, 1, vocab_size]
            probs_t = outputs["probs"].squeeze(1)
            probs[:, t, :] = probs_t

            # sample the next input word and get the corresponding probs
            sampled = self.sample_next_word(probs_t, method, temp)
            w_t = sampled["w_t"]
            sampled_logprobs[:, t] = sampled["probs"]

            # decide whether to stop
            if t >= 1:
                if t == 1:
                    unfinished = w_t != self.end_idx
                else:
                    unfinished = unfinished * (w_t != self.end_idx)
                # w_t[~unfinished] = self.end_idx
                seqs[:, t] = w_t
                seqs[:, t][~unfinished] = self.end_idx
                if unfinished.sum() == 0:
                    break
            else:
                seqs[:, t] = w_t


        sampled = {"seqs": seqs, "probs": probs, "sampled_logprobs": sampled_logprobs}
        return sampled

    def sample_next_word(self, probs, method, temp=1):
        """Sample the next word, given probs output  by rnn
        """
        logprobs = torch.log_softmax(probs, dim=1)
        if method == "greedy":
            sampled_logprobs, w_t = torch.max(logprobs.detach(), 1)
        else:
            prob_prev = torch.exp(logprobs / temp)
            w_t = torch.multinomial(prob_prev, 1)
            # w_t: [N, 1]
            sampled_logprobs = logprobs.gather(1, w_t).squeeze(1)
            w_t = w_t.view(-1)
        w_t = w_t.detach().long()

        # sampled_probs: [N,], w_t: [N,]
        return {"w_t": w_t, "probs": sampled_logprobs}

    def sample_beam(self, encoded, states, **kwargs):
        # encoded: [N, emb_dim]
        # states: [num_layers, N, enc_hid_dim]
        seqs = torch.zeros(encoded.size(0), kwargs["max_length"], dtype=torch.long)
        # beam search can only be used sentence by sentence
        for i in range(encoded.size(0)):
            encoded_i = encoded[i]
            states_i = states if states is None else states[:, i, :]
            seq_i = self.sample_beam_core(encoded_i, states_i, **kwargs)
            seqs[i] = seq_i

        return {"seqs": seqs}

    def sample_beam_core(self, encoded, state, **kwargs):
        """
        Beam search decoding of a single sentence
        Params:
            encoded: [emb_dim,]
            state: [num_layers, enc_hid_dim]
            beam_size: int
        """
        k = kwargs["beam_size"]
        max_length = kwargs["max_length"]
        top_k_probs = torch.zeros(k).to(encoded.device)

        if not state is None:
            state = state.reshape(state.size(0), 1, -1).expand(state.size(0), k, -1)
            state = state.contiguous()
            # state: [num_layers, k, enc_hid_dim]
        
        for t in range(max_length):
            if t == 0:
                e_t = encoded.reshape(1, -1).expand(k, -1)
            else:
                e_t = self.word_embeddings(next_word_inds)

            # e_t: [k, emb_dim]
            e_t = e_t.unsqueeze(1)
            
            # feed to the decoder to get state
            output = self.decoder(e_t, state)
            state = output["states"]
            # state: [num_layers, k, enc_hid_dim]
            # output["probs"]: [k, 1, vocab_size]
            probs_t = output["probs"].squeeze(1)
            probs_t = torch.log_softmax(probs_t, dim=1)
            # probs_t: [k, vocab_size]

            # calculate the joint probability up to the timestep t
            probs_t = top_k_probs.unsqueeze(1).expand_as(probs_t) + probs_t
            
            if t == 0:
                # for the first step, all k seqs will have the same probs
                top_k_probs, top_k_words = probs_t[0].topk(k, 0, True, True)
            else:
                # unroll and find top probs, and their unrolled indices
                top_k_probs, top_k_words = probs_t.view(-1).topk(k, 0, True, True)

            # convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / self.vocab_size  # [k,]
            next_word_inds = top_k_words % self.vocab_size  # [k,]

            # add new words to sequences
            if t == 0:
                seqs = next_word_inds.unsqueeze(1)
                # seqs: [k, 1]
            else:
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # [k, t + 1]

            state = state[:, prev_word_inds, :].contiguous()

        return seqs[0] # since probs have been sorted, the first sequence is the most probabale result


class CaptionSentenceModel(CaptionModel):

    def __init__(self, encoder, decoder, seq_output_size, **kwargs):
        super(CaptionSentenceModel, self).__init__(encoder, decoder, **kwargs)
        self.output_transform = nn.Sequential()
        if decoder.model.hidden_size != seq_output_size:
            self.output_transform = nn.Linear(decoder.model.hidden_size, seq_output_size)

    def _forward(self, *input, **kwargs):
        """Decode audio feature vectors and generates captions.
           With audio feature and captions as input, i.e., teacher forcing
        """
        feats, feat_lens, caps, cap_lens = input
        encoded, states = self.encoder(feats, feat_lens)
        # encoded: [N, emb_dim], the embedding output by the encoder

        # prepare input to the decoder: encoder output + label embeddings
        embeds = self.word_embeddings(caps)
        embeds = self.dropoutlayer(embeds)
        # embeds: [N, max_len, emb_dim]
        embeds = torch.cat((encoded.unsqueeze(1), embeds), 1)
        # embeds: [N, max_len + 1, emb_dim]

        # prepare packed input to the decoder for efficient training (remove padded zeros)
        # audio feature and the first (max_len - 1) word embeddings are packed
        packed = nn.utils.rnn.pack_padded_sequence(
            embeds, cap_lens, batch_first=True)
        output = self.decoder(packed, states, seq_output=True)
        output["seq_outputs"] = self.output_transform(output["seq_outputs"])

        return output

    def sample_core(self, encoded, states, **kwargs):
        # optional keyword arguments
        method = kwargs.get("method", "greedy")
        temp = kwargs.get("temperature", 1.0)
        max_length = kwargs.get("max_length", self.max_length)
        assert method in ("greedy", "sample", "beam"), "unknown sampling method"

        if method == "beam":
            beam_size = kwargs.get("beam_size", 5)
            return self.sample_beam(
                encoded, states, max_length=max_length, beam_size=beam_size)

        N = encoded.size(0)
        lens = torch.zeros(N)
        seqs = torch.zeros(N, max_length, dtype=torch.long).fill_(self.end_idx)
        probs = torch.zeros(N, max_length, self.vocab_size)
        hiddens = torch.zeros(N, max_length, self.decoder.model.hidden_size).to(encoded.device)
        sampled_logprobs = torch.zeros(N, max_length)

        # start sampling
        for t in range(max_length):
            # prepare input word/audio embedding
            if t == 0:
                e_t = encoded
            else:
                e_t = self.word_embeddings(w_t)
                e_t = self.dropoutlayer(e_t)
            # e_t: [N, emb_dim]
            e_t = e_t.unsqueeze(1)

            # feed to the decoder to get states and probs
            outputs = self.decoder(e_t, states, seq_output=True)
            states = outputs["states"]
            # outputs: 
            # "probs": [N, 1, vocab_size]
            # "hidden": [N, 1, hidden_size]
            probs_t = outputs["probs"].squeeze(1)
            probs[:, t, :] = probs_t
            hiddens[:, t, :] = outputs["hidden"].squeeze(1)

            # sample the next input word and get the corresponding probs
            sampled = self.sample_next_word(probs_t, method, temp)
            w_t = sampled["w_t"]
            sampled_logprobs[:, t] = sampled["probs"]

            # decide whether to stop
            if t >= 1:
                if t == 1:
                    unfinished = w_t != self.end_idx
                else:
                    unfinished = unfinished * (w_t != self.end_idx)
                # w_t[~unfinished] = self.end_idx
                seqs[:, t] = w_t
                seqs[:, t][~unfinished] = self.end_idx
                lens[unfinished] += 1
                if unfinished.sum() == 0:
                    break
            else:
                seqs[:, t] = w_t
            
            # obtain sentence outputs
            idxs = torch.arange(max_length, device="cpu").repeat(N).view(N, max_length)
            mask = (idxs < lens.view(-1, 1)).to(encoded.device)
            # mask: [N, T]
            seq_outputs = hiddens * mask.unsqueeze(-1)
            seq_outputs = seq_outputs.sum(1)
            seq_outputs = seq_outputs / lens.unsqueeze(1).to(encoded.device)
            # seq_outputs: [N, E]
            seq_outputs = self.output_transform(seq_outputs)

        output = {"seqs": seqs, "probs": probs, "seq_outputs": seq_outputs,
                  "sampled_logprobs": sampled_logprobs,}
        return output

class SentenceDecoderModel(CaptionModel):

    def __init__(self, encoder, decoder, **kwargs):
        super(SentenceDecoderModel, self).__init__(encoder, decoder, **kwargs)

    def forward(self, *input, **kwargs):
        if len(input) != 3 and len(input) != 1:
            raise Exception("number of input should be either 3 (sent_embeds, caps, cap_lens) or 1 (sent_embeds)!")

        mode = kwargs.get("mode", "forward")
        assert mode in ("forward", "sample"), "unknown running mode"
        # "forward" means teacher forcing training, "sample" means sampling

        if len(input) == 1 and mode == "forward":
            raise Exception("missing caption labels for training!")

        return getattr(self, "_" + mode)(*input, **kwargs)

    def _forward(self, *input, **kwargs):
        """Decode sentence embeddings and generates captions.
           With sentence embeddings and captions as input, i.e., teacher forcing
        """
        sent_embeds, caps, cap_lens = input

        # prepare input to the decoder: encoder output + label embeddings
        embeds = self.word_embeddings(caps)
        embeds = self.dropoutlayer(embeds)
        # embeds: [N, max_len, emb_dim]
        embeds = torch.cat((sent_embeds.unsqueeze(1), embeds), 1)
        # embeds: [N, max_len + 1, emb_dim]

        # prepare packed input to the decoder for efficient training (remove padded zeros)
        # audio feature and the first (max_len - 1) word embeddings are packed
        packed = nn.utils.rnn.pack_padded_sequence(
            embeds, cap_lens, batch_first=True)
        output = self.decoder(packed, None)

        return output

    def _sample(self, *input, **kwargs):
        if len(input) == 3:
            sent_embeds, _, _ = input
        else:
            sent_embeds, = input

        # sent_embeds: [N, emb_dim]
        output = self.sample_core(sent_embeds, None, **kwargs)

        return output
