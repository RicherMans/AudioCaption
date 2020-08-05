# -*- coding: utf-8 -*-

import copy
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
        if isinstance(m, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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


class Block2D(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, x):
        return self.block(x)


class LinearSoftPool(nn.Module):
    """LinearSoftPool
    Linear softmax, takes logits and returns a probability, near to the actual maximum value.
    Taken from the paper:
        A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling
    https://arxiv.org/abs/1810.09050
    """
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, time_decision):
        return (time_decision**2).sum(self.pooldim) / time_decision.sum(
            self.pooldim)


class MeanPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.mean(decision, dim=self.pooldim)

class AttentionPool(nn.Module):  
    """docstring for AttentionPool"""  
    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):  
        super().__init__()  
        self.inputdim = inputdim  
        self.outputdim = outputdim  
        self.pooldim = pooldim  
        self.transform = nn.Linear(inputdim, outputdim)  
        self.activ = nn.Softmax(dim=self.pooldim)  
        self.eps = 1e-7  


    def forward(self, logits, decision):  
        # Input is (B, T, D)  
        # B, T , D  
        w = self.activ(torch.clamp(self.transform(logits), -15, 15))  
        detect = (decision * w).sum(  
            self.pooldim) / (w.sum(self.pooldim) + self.eps)  
        # B, T, D  
        return detect


def parse_poolingfunction(poolingfunction_name='mean', **kwargs):
    """parse_poolingfunction
    A heler function to parse any temporal pooling
    Pooling is done on dimension 1
    :param poolingfunction_name:
    :param **kwargs:
    """
    poolingfunction_name = poolingfunction_name.lower()
    if poolingfunction_name == 'mean':
        return MeanPool(pooldim=1)
    elif poolingfunction_name == 'linear':
        return LinearSoftPool(pooldim=1)
    elif poolingfunction_name == 'attention':  
        return AttentionPool(inputdim=kwargs['inputdim'],  
                             outputdim=kwargs['outputdim'])


class CRNNEncoder(BaseEncoder):

    def __init__(self, inputdim, embed_size, **kwargs):
        super(CRNNEncoder, self).__init__(inputdim, embed_size)
        features = nn.ModuleList()
        self.use_hidden = False
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,
                                                      inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(rnn_input_dim,
                          embed_size // 2,
                          bidirectional=True,
                          batch_first=True)
        self.features.apply(self.init_weights)

    def forward(self, *input):
        x, lens = input
        lens = torch.as_tensor(lens)
        N, T, _ = x.shape
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x, _ = self.gru(x)
        # x = nn.functional.interpolate(
            # x.transpose(1, 2),
            # T,
            # mode='linear',
            # align_corners=False).transpose(1, 2)
        
        lens /= 4
        # x: [N, T, E]
        idxs = torch.arange(x.size(1), device="cpu").repeat(N).view(N, x.size(1))
        mask = (idxs < lens.view(-1, 1)).to(x.device)
        # mask: [N, T]

        x1 = x * mask.unsqueeze(-1)
        x_mean = x1.sum(1) / lens.unsqueeze(1).to(x.device)

        # x2 = x
        # x2[~mask] = float("-inf")
        # x_max, _ = x2.max(1)
        # out = x_mean + x_max

        out = x_mean

        return out, None


class CNN10Encoder(BaseEncoder):

    def __init__(self, inputdim, embed_size, **kwargs):
        super(CNN10Encoder, self).__init__(inputdim, embed_size)
        self.use_hidden = False
        self.features = nn.Sequential(
            Block2D(1, 64),
            Block2D(64, 64),
            nn.LPPool2d(4, (2, 4)),
            Block2D(64, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 2)),
            Block2D(128, 256),
            Block2D(256, 256),
            nn.LPPool2d(4, (1, 2)),
            Block2D(256, 512),
            Block2D(512, 512),
            nn.LPPool2d(4, (1, 2)),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool2d((None, 1)),
        )

        self.temp_pool = parse_poolingfunction(kwargs.get('temppool', 'attention'),
                                               inputdim=512,
                                               outputdim=embed_size)
        self.outputlayer = nn.Linear(512, embed_size)
        self.features.apply(self.init_weights)
        self.outputlayer.apply(self.init_weights)

    def forward(self, *input):
        x, lens = input
        lens = torch.as_tensor(lens)
        batch, time, dim = x.shape
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        decison_time = self.outputlayer(x)
        # decison_time = nn.functional.interpolate(
            # decison_time.transpose(1, 2),
            # time,
            # mode='linear',
            # align_corners=False).transpose(1, 2)
        x = self.temp_pool(x, decison_time).squeeze(1)

        return x, None 


class CNN10CRNNEncoder(BaseEncoder):

    def __init__(self, inputdim, embed_size, crnn, cnn, **kwargs):
        super(CNN10CRNNEncoder, self).__init__(inputdim, embed_size)
        self.use_hidden = False
        self.crnn = crnn
        self.cnn = cnn

    def forward(self, *input):
        crnn_feat, _ = self.crnn(*input)
        cnn_feat, _ = self.cnn(*input)
        # out = (crnn_feat + cnn_feat) / 2
        out = torch.cat((crnn_feat, cnn_feat), dim=-1)
        return out, None


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
        # hid: [num_layers, N, hidden]
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # out: [N, T, hidden]
        if not self.use_hidden:
            hid = None
        if self.representation == 'mean':
            out = out.sum(1)
            lens = lens.reshape(-1, 1).expand(*out.size()).to(out.device)
            out = out / lens.float()
        elif self.representation == 'time':
            indices = (lens - 1).reshape(-1, 1, 1).expand(-1, 1, out.size(-1))
            # indices: [N, 1, hidden]
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


if __name__ == "__main__":
    import os

    state_dict = torch.load(os.path.join(os.getcwd(), "experiments/pretrained_encoder/CNN10.pth"), map_location="cpu")
    encoder = CNN10Encoder(64, 527)

    encoder.load_state_dict(state_dict, strict=False)

    x = torch.randn(4, 1571, 64)

    out = encoder(x, torch.tensor([1571, 1071, 985, 666]))
    print(out[0].shape)
