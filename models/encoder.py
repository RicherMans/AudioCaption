# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    
    """
    Encodes the given input into a fixed sized dimension
    Base encoder class, cannot be called directly
    All encoders should inherit from this class
    """

    def __init__(self, inputdim, embed_size):
        super(BaseEncoder, self).__init__()
        self.inputdim = inputdim
        self.embed_size = embed_size

    def init(self):
        for m in self.modules():
            m.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        raise NotImplementedError


class CNNEncoder(BaseEncoder):

    def __init__(self, inputdim, embed_size, **kwargs):
        """

        :inputdim: TODO
        :embed_size: TODO
        :**kwargs: TODO

        """
        super(CNNEncoder, self).__init__(inputdim, embed_size)
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
        self.init()

    def forward(self, x):
        # Add dimension for filters
        x = x.unsqueeze(1)
        x = self.network(x)
        # Pool the time dimension
        x = x.mean(2)
        x = x.view(x.shape[0], x.shape[1]*x.shape[2])
        return self.outputlayer(x), None


class PreTrainedCNN(BaseEncoder):

    """Model that does not update its layers expect last layer"""

    def __init__(self, inputdim, embed_size, pretrained_model, **kwargs):
        """TODO: to be defined1.

        :inputdim: Input feature dimension
        :embed_size: Output of this module
        :**kwargs: Extra arguments ( config file )

        """
        super(PreTrainedCNN, self).__init__(inputdim, embed_size)

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


class CRNNEncoder(BaseEncoder):

    def __init__(self, inputdim, embed_size, **kwargs):
        """

        :inputdim: TODO
        :embed_size: TODO
        :**kwargs: TODO

        """
        super(CRNNEncoder, self).__init__(inputdim, embed_size)
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
        self.init()

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


class GRUEncoder(BaseEncoder):

    def __init__(self, inputdim, embed_size, **kwargs):
        super(GRUEncoder, self).__init__(inputdim, embed_size)
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
        self.init()

    def forward(self, *input):
        x, lens = input
        lens = torch.as_tensor(lens)
        # x: [N, T, D]
        packed = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        packed_out, hid = self.network(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # out: [N, T, E]
        if not self.use_hidden:
            hid = None
        # TODO maybe something wrong with the use of rnn, when num_layers > 1 / bidirectional = True
        if self.representation == 'mean':
            out = out.sum(1)
            lens = lens.reshape(-1, 1).expand(*out.size()).to(out.device)
            out = out / lens.float()
        elif self.representation == 'time':
            indices = (lens - 1).reshape(-1, 1, 1).expand(-1, 1, out.size(-1))
            # indices: [N, 1, E]
            out = torch.gather(out, 1, indices).squeeze(1)

        return self.bn(self.outputlayer(out)), hid


class LSTMEncoder(BaseEncoder):

    def __init__(self, inputdim, embed_size, **kwargs):
        super(LSTMEncoder, self).__init__(inputdim, embed_size)
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
        self.init()

    def forward(self, *input):
        x, lens = input
        lens = torch.as_tensor(lens)
        # x: [N, T, D]
        packed = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        packed_out, hid = self.network(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # out: [N, T, E]
        if not self.use_hidden:
            hid = None
        # TODO maybe something wrong with the use of rnn, when num_layers > 1 / bidirectional = True
        if self.representation == 'mean':
            out = out.sum(1)
            lens = lens.reshape(-1, 1).expand(*out.size()).to(out.device)
            out = out / lens.float()
        elif self.representation == 'time':
            indices = (lens - 1).reshape(-1, 1, 1).expand(-1, 1, out.size(-1))
            # indices: [N, 1, E]
            out = torch.gather(out, 1, indices).squeeze(1)

        return self.outputlayer(out), hid

