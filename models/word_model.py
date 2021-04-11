# -*- coding: utf-8 -*-

import pdb
import torch
import torch.nn as nn


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

        if hasattr(encoder, "use_hidden") and encoder.use_hidden:
            assert encoder.network.hidden_size == decoder.model.hidden_size, \
                "hidden size not compatible while use hidden!"
            assert encoder.network.num_layers == decoder.model.num_layers, \
                """number of layers not compatible while use hidden!
                please either set use_hidden as False or use the same number of layers"""

        dropout_p = kwargs.get("dropout", 0.0)
        self.embed_size = encoder.embed_size
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
        # input sanity check
        if len(input) != 4 and len(input) != 2:
            raise Exception("number of input should be either 4 (feats, feat_lens, caps, cap_lens) or 2 (feats, feat_lens)!")

        if len(input) == 4:
            feats, feat_lens, caps, cap_lens = input
        else:
            feats, feat_lens = input
            caps = None
            cap_lens = None

        encoded = self.encoder(feats, feat_lens)
        # encoded: {
        #     audio_embeds: [N, emb_dim]
        #     audio_embeds_time: [N, src_max_len, emb_dim]
        #     state: rnn style hidden states, [num_dire * num_layers, N, hs_enc]
        #     audio_embeds_lens: [N, ]
        output = self.sample(encoded, caps, cap_lens, **kwargs)
        return output
    
    def sample(self, encoded, caps, cap_lens, **kwargs):
        # optional keyword arguments
        method = kwargs.get("method", "greedy")
        temp = kwargs.get("temperature", 1.0)
        max_length = kwargs.get("max_length", self.max_length)

        if cap_lens is not None:
            max_length = max(cap_lens)

        assert method in ("greedy", "sample", "beam"), "unknown sampling method"

        if method == "beam":
            beam_size = kwargs.get("beam_size", 5)
            return self.sample_beam(
                encoded, max_length=max_length, beam_size=beam_size)

        audio_embeds = encoded["audio_embeds"]
        h_t = encoded["state"]

        N = audio_embeds.size(0)
        seqs = torch.empty(N, max_length, dtype=torch.long).fill_(self.end_idx)
        logits = torch.empty(N, max_length, self.vocab_size).to(audio_embeds.device)
        sampled_logprobs = torch.zeros(N, max_length)

        # start sampling
        for t in range(max_length):
            # prepare input word/audio embedding
            if t == 0:
                e_t = audio_embeds
            else:
                e_t = self.word_embeddings(w_t)
                e_t = self.dropoutlayer(e_t)
            # e_t: [N, emb_dim]
            e_t = e_t.unsqueeze(1)

            # feed to the decoder to get states and logits
            output_t = self.decoder(e_t, h_t)
            h_t = output_t["states"]

            # outputs["logits"]: [N, 1, vocab_size]
            logits_t = output_t["logits"].squeeze(1)
            logits[:, t, :] = logits_t

            # sample the next input word and get the corresponding logits
            sampled = self.sample_next_word(logits_t, method, temp)
            w_t = sampled["w_t"]
            sampled_logprobs[:, t] = sampled["probs"]
            seqs[:, t] = w_t

            if kwargs["mode"] != "train": # decide whether to stop when sampling
                if t == 0:
                    unfinished = w_t != self.end_idx
                else:
                    unfinished = unfinished * (w_t != self.end_idx)
                seqs[:, t][~unfinished] = self.end_idx
                if unfinished.sum() == 0:
                    break
            else:
                # training
                w_t = caps[:, t]

        output = {
            "seqs": seqs, 
            "logits": logits, 
            "sampled_logprobs": sampled_logprobs
        }
        return output

    def sample_next_word(self, logits, method, temp=1):
        """Sample the next word, given probs output by rnn
        """
        logprobs = torch.log_softmax(logits, dim=1)
        if method == "greedy":
            sampled_logprobs, w_t = torch.max(logprobs.detach(), 1)
        else:
            prob_prev = torch.exp(logprobs / temp)
            w_t = torch.multinomial(prob_prev, 1)
            # w_t: [N, 1]
            sampled_logprobs = logprobs.gather(1, w_t).squeeze(1)
            w_t = w_t.view(-1)
        w_t = w_t.detach().long()

        # sampled_logprobs: [N,], w_t: [N,]
        return {"w_t": w_t, "probs": sampled_logprobs}

    def sample_beam(self, encoded, **kwargs):
        # encoded: {"audio_embeds", "state"}
        audio_embeds = encoded["audio_embeds"]
        states = encoded["state"]
        seqs = torch.zeros(audio_embeds.size(0), kwargs["max_length"], dtype=torch.long)
        # beam search sentence by sentence
        for i in range(audio_embeds.size(0)):
            encoded_i = audio_embeds[i]
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
        top_k_logprobs = torch.zeros(k).to(encoded.device)

        if state is not None:
            state = state.reshape(state.size(0), 1, -1).expand(state.size(0), k, -1)
            state = state.contiguous()
            # state: [num_layers, k, enc_hid_dim]
        
        for t in range(max_length):
            if t == 0:
                e_t = encoded.reshape(1, -1).expand(k, -1)
            else:
                e_t = self.word_embeddings(next_word_inds)
                e_t = self.dropoutlayer(e_t)

            # e_t: [k, emb_dim]
            e_t = e_t.unsqueeze(1)
            
            # feed to the decoder to get state
            output = self.decoder(e_t, state)
            state = output["states"]
            # state: [num_layers, k, enc_hid_dim]
            # output["logits"]: [k, 1, vocab_size]
            logits_t = output["logits"].squeeze(1)
            logprobs_t = torch.log_softmax(logits_t, dim=1)
            # logprobs_t: [k, vocab_size]

            # calculate the joint probability up to the timestep t
            logprobs_t = top_k_logprobs.unsqueeze(1).expand_as(logprobs_t) + logprobs_t
            
            if t == 0:
                # for the first step, all k seqs will have the same probs
                top_k_logprobs, top_k_words = logprobs_t[0].topk(k, 0, True, True)
            else:
                # unroll and find top logprobs, and their unrolled indices
                top_k_logprobs, top_k_words = logprobs_t.view(-1).topk(k, 0, True, True)

            # convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // self.vocab_size  # [k,]
            next_word_inds = top_k_words % self.vocab_size  # [k,]

            # add new words to sequences
            if t == 0:
                seqs = next_word_inds.unsqueeze(1)
                # seqs: [k, 1]
            else:
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # [k, t + 1]

            state = state[:, prev_word_inds, :].contiguous()

        return seqs[0] # since logprobs have been sorted, the first sequence is the most probabale result


class CaptionSentenceModel(CaptionModel):

    def __init__(self, encoder, decoder, seq_output_size, **kwargs):
        super(CaptionSentenceModel, self).__init__(encoder, decoder, **kwargs)
        self.output_transform = nn.Sequential()
        if decoder.model.hidden_size != seq_output_size:
            self.output_transform = nn.Linear(decoder.model.hidden_size, seq_output_size)

    def sample(self, encoded, caps, cap_lens, **kwargs):
        # optional keyword arguments
        method = kwargs.get("method", "greedy")
        temp = kwargs.get("temperature", 1.0)
        max_length = kwargs.get("max_length", self.max_length)
        if cap_lens is not None:
            max_length = max(cap_lens)
            cap_lens = torch.as_tensor(cap_lens)
        assert method in ("greedy", "sample", "beam"), "unknown sampling method"

        if method == "beam":
            beam_size = kwargs.get("beam_size", 5)
            return self.sample_beam(
                encoded, max_length=max_length, beam_size=beam_size)

        audio_embeds = encoded["audio_embeds"]
        h_t = encoded["state"]

        N = audio_embeds.size(0)
        lens = torch.zeros(N)
        seqs = torch.zeros(N, max_length, dtype=torch.long).fill_(self.end_idx)
        logits = torch.zeros(N, max_length, self.vocab_size)
        hiddens = torch.zeros(N, max_length, self.decoder.model.hidden_size).to(audio_embeds.device)
        sampled_logprobs = torch.zeros(N, max_length)

        # start sampling
        for t in range(max_length):
            # prepare input word/audio embedding
            if t == 0:
                e_t = audio_embeds
            else:
                e_t = self.word_embeddings(w_t)
                e_t = self.dropoutlayer(e_t)
            # e_t: [N, emb_dim]
            e_t = e_t.unsqueeze(1)

            # feed to the decoder to get states and logits
            output_t = self.decoder(e_t, h_t, seq_output=True)
            h_t = output_t["states"]
            # outputs: 
            # "logits": [N, 1, vocab_size]
            # "hidden": [N, 1, hidden_size]
            logits_t = output_t["logits"].squeeze(1)
            logits[:, t, :] = logits_t
            hiddens[:, t, :] = output_t["states"].squeeze(1)

            # sample the next input word and get the corresponding logits
            sampled = self.sample_next_word(logits_t, method, temp)
            w_t = sampled["w_t"]
            sampled_logprobs[:, t] = sampled["probs"]

            seqs[:, t] = w_t
            # decide whether to stop
            if t == 0:
                unfinished = w_t != self.end_idx
            else:
                unfinished = unfinished * (w_t != self.end_idx)
            seqs[:, t][~unfinished] = self.end_idx
            lens[unfinished] += 1
            if unfinished.sum() == 0 and caps is not None:
                break

            # teacher forcing training
            if caps is not None:
                w_t = caps[:, t]
            
        # obtain sentence outputs
        idxs = torch.arange(max_length, device="cpu").repeat(N).view(N, max_length)
        # import pdb; pdb.set_trace()
        if cap_lens is not None:
            mask = (idxs < cap_lens.view(-1, 1)).to(audio_embeds.device)
        else:
            mask = (idxs < lens.view(-1, 1)).to(audio_embeds.device)
        # mask: [N, T]
        seq_outputs = hiddens * mask.unsqueeze(-1)
        seq_outputs = seq_outputs.sum(1)
        seq_outputs = seq_outputs / lens.unsqueeze(1).to(audio_embeds.device)
        # seq_outputs: [N, E]
        seq_outputs = self.output_transform(seq_outputs)

        output = {"seqs": seqs, "logits": logits, "seq_outputs": seq_outputs,
                  "sampled_logprobs": sampled_logprobs}
        return output

