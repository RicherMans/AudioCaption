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
        
        self.word_embeddings = nn.Embedding(self.vocab_size, embeddings.shape[1])
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
        return probs

    def _sample(self, *input, **kwargs):
        if len(input) == 4:
            feats, feat_lens, _, _ = input
        else:
            feats, feat_lens = input

        # feats: [N, T, F]
        encoded, states = self.encoder(feats, feat_lens)
        # encoded: [N, emb_dim]
        # states: [num_layers, N, enc_hid_dim]
        sampled = self.sample_core(encoded, states, **kwargs)

        return sampled
    
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


class StModel(CaptionModel):

    def __init__(self, encoder, decoder, vocabulary, **kwargs):
        super(StModel, self).__init__(encoder, decoder, **kwargs)
        self.vocabulary = vocabulary

    def forward(self, *input, **kwargs):
        if len(input) != 4 and len(input) != 2:
            raise Exception("number of input should be either 4 (feats, feat_lens, caps, cap_lens) or 2 (feats, feat_lens)!")

        mode = kwargs.get("mode", "forward")
        assert mode in ("forward", "sample", "st"), "unknown running mode"
        # "forward": teacher forcing training
        # "sample": sampling
        # "st": sequence training

        if len(input) == 2 and mode in ("forward", "st"):
            raise Exception("missing caption labels for training!")

        return getattr(self, "_" + mode)(*input, **kwargs)

    def _st(self, *input, **kwargs):
        output = {}

        sample_kwargs = {
            "temperature": kwargs.get("temperature", 1.0),
            "max_length": kwargs.get("max_length", self.max_length)
        }

        feats, feat_lens, caps, cap_lens = input
        # feats: [N, T, F]
        encoded, states = self.encoder(feats, feat_lens)
        # encoded: [N, emb_dim]
        # states: [num_layers, N, enc_hid_dim]

        sampled = self.sample_core(encoded, states, method="sample", **sample_kwargs)
        output["sampled_seqs"] = sampled["seqs"]

        score_kwargs = {
            "N": kwargs.get("bleu_number", 4),
            "smoothing": kwargs.get("smoothing", "method1")
        }
        score = self.get_reward(sampled["seqs"],
                                caps,
                                **score_kwargs)
        # score: [N, ]
        output["score"] = torch.as_tensor(score)

        reward = np.repeat(score[:, np.newaxis], sampled["seqs"].size(-1), 1)
        reward = torch.as_tensor(reward).float()
        mask = (sampled["seqs"] != self.end_idx).float()
        mask = torch.cat([torch.ones(mask.size(0), 1), mask[:, :-1]], 1)
        mask = torch.as_tensor(mask).float()
        loss = - sampled["sampled_logprobs"] * reward * mask
        loss = loss.to(feats.device)
        # loss: [N, max_length]
        loss = torch.sum(loss, dim=1).mean()
        # loss = torch.sum(loss) / torch.sum(mask)
        output["loss"] = loss

        return output

    def get_reward(self, sampled_seqs, gts, **kwargs):
        # sampled_seqs, gts: [N, max_length]
        sampled_seqs = sampled_seqs.cpu().numpy()
        gts = gts.cpu().numpy()

        sampled_score = score_util.compute_bleu_score(sampled_seqs,
                                                      gts,
                                                      self.start_idx,
                                                      self.end_idx,
                                                      self.vocabulary,
                                                      **kwargs)
        # sampled_score = score_util.compute_bert_score(sampled_seqs,
                                                      # gts,
                                                      # self.start_idx,
                                                      # self.end_idx,
                                                      # self.vocabulary,
                                                      # **kwargs)
        return sampled_score


class ScstModel(StModel):

    def __init__(self, encoder, decoder, vocabulary, **kwargs):
        super(ScstModel, self).__init__(encoder, decoder, vocabulary, **kwargs)

    def forward(self, *input, **kwargs):
        if len(input) != 4 and len(input) != 2:
            raise Exception("number of input should be either 4 (feats, feat_lens, caps, cap_lens) or 2 (feats, feat_lens)!")

        mode = kwargs.get("mode", "forward")
        assert mode in ("forward", "sample", "scst"), "unknown running mode"
        # "forward": teacher forcing training
        # "sample": sampling
        # "scst": self-critical sequence training

        if len(input) == 2 and mode in ("forward", "scst"):
            raise Exception("missing caption labels for training!")

        return getattr(self, "_" + mode)(*input, **kwargs)

    def _scst(self, *input, **kwargs):
        output = {}

        sample_kwargs = {
            "temperature": kwargs.get("temperature", 1.0),
            "max_length": kwargs.get("max_length", self.max_length)
        }

        feats, feat_lens, caps, cap_lens = input
        # feats: [N, T, F]
        encoded, states = self.encoder(feats, feat_lens)
        # encoded: [N, emb_dim]
        # states: [num_layers, N, enc_hid_dim]

        # prepare baseline
        self.eval()
        with torch.no_grad():
            sampled_greedy = self.sample_core(
                encoded, states, method="greedy", **sample_kwargs)
        output["greedy_seqs"] = sampled_greedy["seqs"]

        self.train()
        sampled = self.sample_core(encoded, states, method="sample", **sample_kwargs)
        output["sampled_seqs"] = sampled["seqs"]

        score_kwargs = {
            "N": kwargs.get("bleu_number", 4),
            "smoothing": kwargs.get("smoothing", "method1")
        }
        reward_score = self.get_self_critical_reward(sampled_greedy["seqs"],
                                                     sampled["seqs"],
                                                     caps,
                                                     **score_kwargs)
        # reward: [N, ]
        output["reward"] = torch.as_tensor(reward_score["reward"])
        output["score"] = torch.as_tensor(reward_score["score"])

        reward = np.repeat(reward_score["reward"][:, np.newaxis], sampled["seqs"].size(-1), 1)
        reward = torch.as_tensor(reward).float()
        mask = (sampled["seqs"] != self.end_idx).float()
        mask = torch.cat([torch.ones(mask.size(0), 1), mask[:, :-1]], 1)
        mask = torch.as_tensor(mask).float()
        loss = - sampled["sampled_logprobs"] * reward * mask
        loss = loss.to(feats.device)
        # loss: [N, max_length]
        loss = torch.sum(loss, dim=1).mean()
        # loss = torch.sum(loss) / torch.sum(mask)
        output["loss"] = loss

        return output

    def get_self_critical_reward(self, greedy_seqs, sampled_seqs, gts, **kwargs):
        # greedy_seqs, sampled_seqs, gts: [N, max_length]
        greedy_seqs = greedy_seqs.cpu().numpy()
        sampled_seqs = sampled_seqs.cpu().numpy()
        gts = gts.cpu().numpy()

        sampled_score = score_util.compute_bleu_score(sampled_seqs,
                                                      gts,
                                                      self.start_idx,
                                                      self.end_idx,
                                                      self.vocabulary,
                                                      **kwargs)
        greedy_score = score_util.compute_bleu_score(greedy_seqs, 
                                                     gts,
                                                     self.start_idx,
                                                     self.end_idx,
                                                     self.vocabulary,
                                                     **kwargs)
        # sampled_score = score_util.compute_bert_score(sampled_seqs,
                                                      # gts,
                                                      # self.start_idx,
                                                      # self.end_idx,
                                                      # self.vocabulary,
                                                      # **kwargs)
        # greedy_score = score_util.compute_bert_score(greedy_seqs,
                                                     # gts,
                                                     # self.start_idx,
                                                     # self.end_idx,
                                                     # self.vocabulary,
                                                     # **kwargs)
        reward = sampled_score - greedy_score
        return {"reward": reward, "score": sampled_score}

class MixerModel(StModel):

    def __init__(self, encoder, decoder, vocabulary, **kwargs):
        super(MixerModel, self).__init__(encoder, decoder, vocabulary, **kwargs)
        self.rl_steps = kwargs.get("init_rl_steps", 2)
        self.XE_criterion = nn.CrossEntropyLoss()

    def _st(self, *input, **kwargs):
        output = {}

        max_length = kwargs.get("max_length", self.max_length)
        sample_kwargs = {
            "temperature": kwargs.get("temperature", 1.0),
            "max_length": max_length
        }

        feats, feat_lens, caps, cap_lens = input
        # feats: [N, T, F]
        encoded, states = self.encoder(feats, feat_lens)
        # encoded: [N, emb_dim]
        # states: [num_layers, N, enc_hid_dim]

        sampled = self.sample_core(encoded, states, method="sample", **sample_kwargs)
        output["sampled_seqs"] = sampled["seqs"]

        XE_lens = np.maximum(cap_lens - self.rl_steps, 0)
        # probs = nn.utils.rnn.pack_padded_sequence(sampled["probs"], XE_lens, batch_first=True).data
        # probs = probs.to(feats.device)
        # targets = nn.utils.rnn.pack_padded_sequence(caps, XE_lens, batch_first=True).data
        if XE_lens.sum() == 0:
            output["XE_loss"] = 0
        else:
            probs = []
            targets = []
            for i in range(caps.size(0)):
                probs.append(sampled["probs"][i, :XE_lens[i]])
                targets.append(caps[i, :XE_lens[i]])
            probs = torch.cat(probs).to(feats.device)
            targets = torch.cat(targets).to(feats.device)
            XE_loss = self.XE_criterion(probs, targets)
            output["XE_loss"] = XE_loss

        score_kwargs = {
            "N": kwargs.get("bleu_number", 4),
            "smoothing": kwargs.get("smoothing", "method1")
        }
        score = self.get_reward(sampled["seqs"],
                                caps,
                                **score_kwargs)
        # score: [N, ]
        output["score"] = torch.as_tensor(score)
        reward = np.repeat(score[:, np.newaxis], sampled["seqs"].size(-1), 1)
        reward = torch.as_tensor(reward).float()
        rl_mask = np.zeros_like(caps.cpu())
        for i in range(caps.size(0)):
            rl_mask[i, XE_lens[i]: cap_lens[i]] = 1
        rl_mask = torch.as_tensor(rl_mask).float()
        rl_loss = - sampled["sampled_logprobs"] * reward * rl_mask
        rl_loss = rl_loss.to(feats.device)
        rl_loss = rl_loss.sum() / rl_mask.sum()
        output["rl_loss"] = rl_loss
        
        return output


class SentenceModel(CaptionModel):

    def __init__(self, encoder, decoder, 
                 vocabulary, sent_dim, **kwargs):
        super(encoder, decoder, vocabulary, **kwargs)
        self.sent_dim = sent_dim
        if self.decoder.model.hidden_size != sent_dim:
            self.sent_projection = nn.Linear(sent_dim,
                                             self.decoder.model.hidden_size)
        else:
            self.sent_projection = nn.Sequential()

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
         
        return self.decoder(packed, states)

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
