# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

from models.WordModel import CaptionModel
from utils import score_util


class StModel(CaptionModel):

    def __init__(self, encoder, decoder, vocabulary, **kwargs):
        super(StModel, self).__init__(encoder, decoder, **kwargs)
        self.vocabulary = vocabulary

    def forward(self, *input, **kwargs):
        if len(input) != 4 and len(input) != 2:
            raise Exception(
                """number of input should be either 4 (feats, feat_lens, keys, key2caps) for 'st'
                                                      (feats, feat_lens, caps, cap_lens) for 'forward'
                                                 or 2 (feats, feat_lens) for 'sample'!""")

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

        feats, feat_lens, keys, key2caps = input
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
                                keys,
                                key2caps,
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

    def get_reward(self, sampled_seqs, keys, key2caps, **kwargs):
        # sampled_seqs, gts: [N, max_length]
        sampled_seqs = sampled_seqs.cpu().numpy()
        gts = gts.cpu().numpy()

        # sampled_score = score_util.compute_bleu_score(sampled_seqs,
                                                      # gts,
                                                      # self.start_idx,
                                                      # self.end_idx,
                                                      # self.vocabulary,
                                                      # **kwargs)
        sampled_score = score_util.compute_cider_score(sampled_seqs,
                                                       keys,
                                                       key2caps,
                                                       self.start_idx,
                                                       self.end_idx,
                                                       self.vocabulary)
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
        """Decode audio feature vectors and generates captions.
        """
        if len(input) != 4 and len(input) != 2:
            raise Exception("number of input should be either 4 (feats, feat_lens, keys, caps) or 2 (feats, feat_lens)!")

        if len(input) == 4:
            feats, feat_lens, keys, caps = input
        else:
            feats, feat_lens = input
            caps = None
            cap_lens = None
        encoded = self.encoder(feats, feat_lens)
        # encoded: 
        #     audio_embeds: [N, emb_dim]
        #     audio_embeds_time: [N, src_max_len, emb_dim]
        #     state: rnn style encoder states
        #     audio_embeds_lens: [N, ]
        if len(input) == 2:
            output = self.sample(encoded, None, None, **kwargs)
        else:
            output = self.scst(encoded, keys, caps, **kwargs)

        return output

    def scst(self, encoded, keys, refs, **kwargs):
        output = {}

        sample_kwargs = {
            "temperature": kwargs.get("temperature", 1.0),
            "max_length": kwargs["max_length"]
        }

        # prepare baseline
        self.eval()
        with torch.no_grad():
            sampled_greedy = self.sample(
                encoded, None, None, method="greedy", **sample_kwargs)
        output["greedy_seqs"] = sampled_greedy["seqs"]

        self.train()
        sampled = self.sample(encoded, None, None, method="sample", **sample_kwargs)
        output["sampled_seqs"] = sampled["seqs"]

        refs = refs.cpu().numpy()
        reward_score = self.get_self_critical_reward(sampled_greedy["seqs"],
                                                     sampled["seqs"],
                                                     keys,
                                                     # key2refs,
                                                     refs,
                                                     kwargs["scorer"])
        # reward: [N, ]
        output["reward"] = torch.as_tensor(reward_score["reward"])
        output["score"] = torch.as_tensor(reward_score["score"])

        reward = np.repeat(reward_score["reward"][:, np.newaxis], sampled["seqs"].size(-1), 1)
        reward = torch.as_tensor(reward).float()
        mask = (sampled["seqs"] != self.end_idx).float()
        mask = torch.cat([torch.ones(mask.size(0), 1), mask[:, :-1]], 1)
        mask = torch.as_tensor(mask).float()
        loss = - sampled["sampled_logprobs"] * reward * mask
        loss = loss.to(encoded["audio_embeds"].device)
        # loss: [N, max_length]
        loss = torch.sum(loss, dim=1).mean()
        # loss = torch.sum(loss) / torch.sum(mask)
        output["loss"] = loss

        return output

    def get_self_critical_reward(self, greedy_seqs, sampled_seqs, 
                                 keys, refs, scorer):
        # greedy_seqs, sampled_seqs: [N, max_length]
        greedy_seqs = greedy_seqs.cpu().numpy()
        sampled_seqs = sampled_seqs.cpu().numpy()

        sampled_score = score_util.compute_batch_score(sampled_seqs,
                                                       # key2refs,
                                                       refs,
                                                       keys,
                                                       self.start_idx,
                                                       self.end_idx,
                                                       self.vocabulary,
                                                       scorer)
        greedy_score = score_util.compute_batch_score(greedy_seqs, 
                                                      # key2refs,
                                                      refs,
                                                      keys,
                                                      self.start_idx,
                                                      self.end_idx,
                                                      self.vocabulary,
                                                      scorer)
        # sampled_score = score_util.compute_cider_score(sampled_seqs,
                                                       # keys,
                                                       # key2refs,
                                                       # self.start_idx,
                                                       # self.end_idx,
                                                       # self.vocabulary)
        # greedy_score = score_util.compute_cider_score(greedy_seqs, 
                                                      # keys,
                                                      # key2refs,
                                                      # self.start_idx,
                                                      # self.end_idx,
                                                      # self.vocabulary)
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
