# -*- coding: utf-8 -*-
# @Author: richman
# @Date:   2018-03-28
# @Last Modified by:   richman
# @Last Modified time: 2018-03-28

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):

    """Encodes the given input into a fixed sized dimension"""

    def __init__(self, inputdim, embed_size, **kwargs):
        """

        :inputdim: TODO
        :embed_size: TODO
        :**kwargs: TODO

        """
        super(CNNEncoder, self).__init__()
        self._inputdim = inputdim
        self._embed_size = embed_size
        self._filtersizes = kwargs.get('filtersizes', [5, 3, 3])
        self._filter = kwargs.get('filter', [32, 32, 32])
        self._filter = [1] + self._filter
        net = nn.ModuleList()
        for nl, (h0, h1, filtersize) in enumerate(
                zip(self._filter, self._filter[1:], self._filtersizes)):
            if nl > 0:
                # GLU Output halves
                h0 = h0//2
            net.append(
                nn.Conv2d(
                    h0,
                    h1,
                    filtersize,
                    padding=int(
                        filtersize /
                        2),
                    bias=False))
            net.append(nn.BatchNorm2d(h1))
            net.append(nn.GLU(dim=1))
            net.append(nn.MaxPool2d((1, 2)))
        self.network = nn.Sequential(*net)

        def calculate_size(input_size):
            x = torch.randn(input_size).unsqueeze(0)
            output = self.network(x)
            return output.size()[1:]
        outputdim = calculate_size((1, 500, inputdim))[-1]
        self.outputlayer = nn.Linear(
            self._filter[-1]//2 * outputdim, self._embed_size)
        self.network.apply(self.init_weights)
        self.outputlayer.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        # Add dimension for filters
        x = x.unsqueeze(1)
        x = self.network(x)
        # Pool the time dimension
        x = x.mean(2)
        x = x.view(x.shape[0], x.shape[1]*x.shape[2])
        return self.outputlayer(x), None


class PreTrainedCNN(nn.Module):

    """Model that does not update its layers expect last layer"""

    def __init__(self, inputdim, embed_size, pretrained_model, **kwargs):
        """TODO: to be defined1.

        :inputdim: Input feature dimension
        :embed_size: Output of this module
        :**kwargs: Extra arguments ( config file )

        """
        nn.Module.__init__(self)

        self._inputdim = inputdim
        self._embed_size = embed_size
        # Remove last output layer
        modules = list(pretrained_model.children())[:-1]
        self.network = nn.Sequential(*modules)

        def calculate_size(input_size):
            x = torch.randn(input_size).unsqueeze(0)
            output = pretrained_model.network(x)
            return output.size()[1:]
        outputdim = calculate_size((1, 500, inputdim))[-1]//2
        self.outputlayer = nn.Linear(
            outputdim * pretrained_model._filter[-1],
            embed_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        with torch.no_grad():
            x = self.network(x)
            x = x.mean(2)
            x = x.view(x.shape[0], x.shape[1]*x.shape[2])
        return self.outputlayer(x), None


class CRNNEncoder(nn.Module):

    """Encodes the given input into a fixed sized dimension"""

    def __init__(self, inputdim, embed_size, **kwargs):
        """

        :inputdim: TODO
        :embed_size: TODO
        :**kwargs: TODO

        """
        super(CRNNEncoder, self).__init__()
        self._inputdim = inputdim
        self._embed_size = embed_size
        self._filtersizes = kwargs.get('filtersizes', [5, 3, 3])
        self._filter = kwargs.get('filter', [32, 32, 32])
        self._hidden_size = kwargs.get('hidden_size', 512)
        self._bidirectional = kwargs.get('bidirectional', False)
        self._filter = [1] + self._filter
        net = nn.ModuleList()
        for nl, (h0, h1, filtersize) in enumerate(
                zip(self._filter, self._filter[1:], self._filtersizes)):
            if nl > 0:
                # GLU Output halves
                h0 = h0//2
            net.append(
                nn.Conv2d(
                    h0,
                    h1,
                    filtersize,
                    padding=int(
                        filtersize /
                        2),
                    bias=False))
            net.append(nn.BatchNorm2d(h1))
            net.append(nn.GLU(dim=1))
            net.append(nn.Dropout2d(0.3))
            net.append(nn.MaxPool2d((1, 2)))
        self.network = nn.Sequential(*net)

        def calculate_size(input_size):
            x = torch.randn(input_size).unsqueeze(0)
            output = self.network(x)
            return output.size()[1:]
        outputdim = calculate_size((1, 500, inputdim))
        self.rnn1 = nn.GRU(
            self._filter[-1] // 2 * outputdim[-1],
            self._hidden_size, bidirectional=self._bidirectional )
        self.rnn2 = nn.GRU(
            self._filter[-1] // 2 * outputdim[-1],
            self._hidden_size, bidirectional=self._bidirectional)
        self.outputlayer = nn.Linear(
            self._hidden_size * (self._bidirectional + 1),
            self._embed_size)
        self.network.apply(self.init_weights)
        self.outputlayer.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        # Add dimension for filters
        x = x.unsqueeze(1)
        x = self.network(x)
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1)
        x1, _ = self.rnn1(x)
        x2, _ = self.rnn2(x)
        x = x1 * torch.sigmoid(x2)
        # # Pool the time dimension
        x = x.mean(1)
        return self.outputlayer(x), None


class GRUEncoder(nn.Module):
    def __init__(self, inputdim, embed_size, **kwargs):
        super(GRUEncoder, self).__init__()
        hidden_size = kwargs.get('hidden_size', 256)
        bidirectional = kwargs.get('bidirectional', False)
        num_layers = kwargs.get('num_layers', 1)
        dropout = kwargs.get('dropout', 0.3)
        self.representation = kwargs.get('representation', 'time')
        assert self.representation in ('time', 'mean')
        self.use_hidden = kwargs.get('use_hidden', False)
        self.network = nn.GRU(
            inputdim,
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True)
        self.outputlayer = nn.Linear(
            hidden_size * (bidirectional + 1), embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        nn.init.kaiming_uniform_(self.outputlayer.weight)

    def forward(self, x):
        x, hid = self.network(x)
        if not self.use_hidden:
            hid = None
        if self.representation == 'mean':
            x = x.mean(1)
        elif self.representation == 'time':
            x = x[:, -1, :]
        return self.bn(self.outputlayer(x)), hid


class GRUDecoder(nn.Module):

    def __init__(self, embed_size, vocab_size, **kwargs):
        super(GRUDecoder, self).__init__()
        self.hidden_size = kwargs.get('hidden_size', 256)
        self.num_layers = kwargs.get('num_layers', 1)
        bidirectional = kwargs.get('bidirectional', False)
        self.batch_first = kwargs.get('batch_first', True)
        dropout_p = kwargs.get('dropout', 0.0)
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.dropoutlayer = nn.Dropout(dropout_p)
        self.model = nn.GRU(
            input_size=embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            bidirectional=bidirectional)
        self.classifier = nn.Linear(
            self.hidden_size * (bidirectional + 1),
            vocab_size)
        nn.init.kaiming_uniform_(self.classifier.weight)
        nn.init.kaiming_uniform_(self.word_embeddings.weight)

    def forward(self, features, captions, lengths, state=None):
        """Decode image feature vectors and generates captions."""
        embeddings = self.word_embeddings(captions)
        # Dropout can be noop in case probability is 0
        embeddings = self.dropoutlayer(embeddings)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True)
        hiddens, _ = self.model(packed, state)
        # padded_vals = nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True, padding_value= -100)
        # print(padded_vals)
        padded_data, lengths = nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True)
        batch_size = len(lengths)
        sentence_ouputs = torch.zeros(batch_size, padded_data.shape[-1])
        for i in range(batch_size):
            length = lengths[i]
            sentence_ouputs[i] = torch.mean(padded_data[i, :length, :], dim=0)
        words_outputs = self.classifier(hiddens[0])
        return words_outputs, sentence_ouputs


    def sample(self, feature, states=None, maxlength=20, return_probs=False):
        sampled_token_ids = []
        feature = feature.unsqueeze(1)
        sampled_probs = []
        for i in range(maxlength):
            hiddens, states = self.model(feature, states)
            # Only a single timestep here, squeeze that dimension
            outputs = self.classifier(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_token_ids.append(predicted)
            sampled_probs.append(outputs)
            predicted = predicted.detach()
            # Prepare next input
            feature = self.word_embeddings(predicted).unsqueeze(1)

        sampled_token_ids = torch.stack(sampled_token_ids, 1)
        if return_probs:
            sampled_probs = torch.stack(sampled_probs, 1)
            return sampled_token_ids, sampled_probs
        return sampled_token_ids


class LSTMEncoder(nn.Module):
    def __init__(self, inputdim, embed_size, **kwargs):
        super(LSTMEncoder, self).__init__()
        hidden_size = kwargs.get('hidden_size', 256)
        bidirectional = kwargs.get('bidirectional', False)
        num_layers = kwargs.get('num_layers', 1)
        self.representation = kwargs.get('representation', 'time')
        assert self.representation in ('time', 'mean')
        self.use_hidden = kwargs.get('use_hidden', True)
        self.network = nn.LSTM(
            inputdim,
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True)
        self.outputlayer = nn.Linear(
            hidden_size * (bidirectional + 1), embed_size)

    def forward(self, x):
        x, hid = self.network(x)
        if not self.use_hidden:
            hid = None
        if self.representation == 'mean':
            x = x.mean(1)
        elif self.representation == 'time':
            x = x[:, -1, :]
        return self.outputlayer(x), hid


class LSTMDecoder(nn.Module):

    def __init__(self, embed_size, vocab_size, **kwargs):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = kwargs.get('hidden_size', 256)
        self.num_layers = kwargs.get('num_layers', 1)
        bidirectional = kwargs.get('bidirectional', False)
        self.batch_first = kwargs.get('batch_first', True)
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.model = nn.LSTM(
            input_size=embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            bidirectional=bidirectional)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * (bidirectional + 1), vocab_size))

    def forward(self, features, captions, lengths, state=None):
        """Decode image feature vectors and generates captions."""
        embeddings = self.word_embeddings(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True)
        hiddens, _ = self.model(packed, state)
        outputs = self.classifier(hiddens[0])
        return outputs

    def sample(self, feature, states=None, maxlength=20, return_probs=False):
        sampled_token_ids = []
        feature = feature.unsqueeze(1)
        sampled_probs = []
        for i in range(maxlength):
            hiddens, states = self.model(feature, states)
            # Only a single timestep here, squeeze that dimension
            outputs = self.classifier(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_token_ids.append(predicted)
            sampled_probs.append(outputs)
            # Prepare next input
            feature = self.word_embeddings(predicted).unsqueeze(1)
        sampled_token_ids = torch.stack(sampled_token_ids, 1)
        if return_probs:
            sampled_probs = torch.stack(sampled_probs, 1)
            return sampled_token_ids, sampled_probs
        return sampled_token_ids
