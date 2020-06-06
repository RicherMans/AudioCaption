import torch
import torch.nn as nn
import numpy as np 

import os
import sys

from utils.score_util import compute_bleu_score, compute_bert_score


class LossWrapper(nn.Module):

    def __init__(self, encoder_model, decoder_model, vocab):
        super(LossWrapper, self).__init__()
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.vocabulary = vocab
        self.end_idx = vocab.word2idx["<end>"]
        self.start_idx = vocab.word2idx["<start>"]
        self.ce_criterion = nn.CrossEntropyLoss()


    def forward(self, features, captions, lengths, scst=True, **kwargs):
        teacher_forcing = kwargs.get("teacher_forcing", True)
        sample_length = kwargs.get("sample_length", 30)
        bert_reward_weight = kwargs.get("bert_reward_weight", 1.0)
        bleu_reward_weight = 1 - bert_reward_weight
        encoded_features, encoder_state = self.encoder_model(features)
        if not scst:
            # trained by normal ce loss
            if teacher_forcing:             
                word_outputs, _ = self.decoder_model(
                    encoded_features, captions, lengths, state=encoder_state)
                targets = nn.utils.rnn.pack_padded_sequence(
                    captions, lengths, batch_first=True).data
                # targets: [total_length,] (without padding)
                loss = self.ce_criterion(word_outputs, targets)
                return loss, word_outputs
            else:
                losses = []
                probs = [] 
                for idx in range(len(encoded_features)):
                    outputs_i, probs_i = self.decoder_model.sample_greedy(
                        encoded_features[idx].unsqueeze(0),
                        states=encoder_state, maxlength=lengths[idx],
                        return_probs=True
                    )
                    probs_i = probs_i.squeeze(0)
                    target_trimmed = captions[idx][1:lengths[idx]]
                    loss = self.ce_criterion(probs_i, target_trimmed)
                    losses.append(loss.reshape(1))
                    probs.append(probs_i)
                return torch.mean(torch.cat(losses)), torch.cat(probs, 0)
        else:
            # self critical
            self.decoder_model.eval()
            with torch.no_grad():
                greedy_res, _ = self.decoder_model.sample(
                    feature=encoded_features,
                    states=encoder_state,
                    maxlength=lengths[0],
                    sample_method="greedy",
                    **kwargs
                )
            self.decoder_model.train()
            sample_res, sample_logprobs = self.decoder_model.sample(
                feature=encoded_features,
                states=encoder_state,
                maxlength=lengths[0],
                sample_method="sample",
                **kwargs
            )
            reward = self.get_self_critical_reward(greedy_res, sample_res, captions, bert_reward_weight=bert_reward_weight, bleu_reward_weight=bleu_reward_weight)
            reward = torch.as_tensor(reward).float().to(features.device)
            loss = self.reward_criterion(sample_logprobs, sample_res.detach(), reward) 
            return loss


    def sample(self, features, sample_length=30, **kwargs):
        temperature = kwargs.get("temperature", 1.0)
        encoded_features, encoder_state = self.encoder_model(features)
        outputs, _ = self.decoder_model.sample(
            encoded_features, states=encoder_state, maxlength=sample_length,
            temperature=temperature, sample_method="sample"
        )
        return outputs


    def get_self_critical_reward(self, greedy_res, sample_res, gts, **kwargs):
        # greedy_res, sample_res: [N, sample_length]
        bert_reward_weight = kwargs.get("bert_reward_weight", 1.0)
        bleu_reward_weight = kwargs.get("bleu_reward_weight", 0.0)

        greedy_res = greedy_res.cpu().numpy()
        sample_res = sample_res.cpu().numpy()

        gts = gts.cpu().numpy()

        if bert_reward_weight > 0:
            sample_score = compute_bert_score(
                sample_res, gts, start_idx=self.start_idx, end_idx=self.end_idx,
                padding_val=0, vocabulary=self.vocabulary)
            greedy_score = compute_bert_score(
                greedy_res, gts, start_idx=self.start_idx, end_idx=self.end_idx,
                padding_val=0, vocabulary=self.vocabulary)
            bert_scores = sample_score - greedy_score
        else:
            bert_scores = 0

        if bleu_reward_weight > 0:
            sample_score = compute_bleu_score(
                sample_res, gts, self.start_idx, self.end_idx)
            greedy_score = compute_bleu_score(
                greedy_res, gts, self.start_idx, self.end_idx)
            bleu_scores = sample_score - greedy_score
        else:
            bleu_scores = 0


        scores = bert_reward_weight * bert_scores + bleu_reward_weight * bleu_scores
        rewards = np.repeat(scores[:, np.newaxis], sample_res.shape[1], 1)

        return rewards


    def reward_criterion(self, logprobs, seq, reward):
        logprobs = logprobs.contiguous().view(-1)
        reward = reward.contiguous().view(-1).to(logprobs.device)
        mask = (seq != self.end_idx).float().contiguous().view(-1).to(logprobs.device)
        loss = - logprobs * reward * mask
        loss = torch.sum(loss) / torch.sum(mask)
        return loss
