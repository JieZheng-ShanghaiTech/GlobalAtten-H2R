#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/16
# @Author  : Eric
# @File    : models_revised.py
# @Software: PyCharm

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as stop


def batch_product(input_rnn_output, mat2):
    # function to accomplish attention
    result = None
    for i in range(input_rnn_output.size()[0]):
        # calculate h*w in each batch, input_rnn_output.size(0)=10
        op = torch.mm(input_rnn_output[i], mat2)
        # op = torch.tanh(op)  # added by eric 2021.1.15
        op = op.unsqueeze(0)
        if result is None:
            result = op
        else:
            result = torch.cat((result, op), 0)
    return result.squeeze(2)


def batch_product_luong(hidden, encoder_output):
    [_, batch, hidden_size] = hidden.size()
    hidden = torch.reshape(hidden, (-1, batch, hidden_size * 2)).squeeze()
    result = None
    for i in range(len(encoder_output)):
        op = hidden * encoder_output[i]
        op = op.unsqueeze(1)
        if result is None:
            result = op
        else:
            result = torch.cat((result, op), 1)
    return result


class RecAttention(nn.Module):
    # attention with bin context vector per HM and HM context vector
    def __init__(self, hm, args):
        super(RecAttention, self).__init__()
        self.num_directions = 2 if args.bidirectional else 1
        if hm == False:
            self.bin_rep_size = args.bin_rnn_size * self.num_directions
        else:
            self.bin_rep_size = args.bin_rnn_size

        self.bin_context_vector = nn.Parameter(torch.Tensor(self.bin_rep_size, 1), requires_grad=True)
        self.soft_max = nn.Softmax(dim=1)
        self.bin_context_vector.data.uniform_(-0.1, 0.1)
        # self.attention_linear = nn.Linear(self.bin_rep_size, self.bin_rep_size)
        # nn.init.xavier_uniform_(self.bin_context_vector.data)
        # nn.init.kaiming_uniform_(self.bin_context_vector.data)

    def forward(self, input_rnn_output):
        # alpha1 [10, 100], alpha2[10, 5]
        # alpha = F.softmax(batch_product(input_rnn_output, self.bin_context_vector))
        # print('bp:', batch_product(input_rnn_output, self.bin_context_vector).shape)
        alpha = self.soft_max(batch_product(input_rnn_output, self.bin_context_vector))  # softmax(h_t * W)
        # u = self.attention_linear(input_rnn_output)
        # print('u', u.shape)
        # alpha = self.soft_max(u)
        # print('alpha:', alpha.shape)
        [batch_size, source_length, bin_rep_size2] = input_rnn_output.size()
        repres = torch.bmm(alpha.unsqueeze(2).view(batch_size, -1, source_length), input_rnn_output)  # [10, 1, 64] [10, 1, 32]
        # print('alpha shape:', alpha.shape)
        # print('repres shape:', repres.shape)
        return repres, alpha


class Recurrent_encoder1(nn.Module):
    # modular LSTM encoder
    def __init__(self, n_bins, ip_bin_size, hm, args):
        super(Recurrent_encoder1, self).__init__()
        self.bin_rnn_size = args.bin_rnn_size  # hidden size
        self.ipsize = ip_bin_size  # ipsize is input_size,  dimension of a word
        self.seq_length = n_bins
        self.num_directions = 2 if args.bidirectional else 1

        if (hm == False):
            self.bin_rnn_size = args.bin_rnn_size
        else:
            self.bin_rnn_size = args.bin_rnn_size // 2

        self.bin_rep_size = self.bin_rnn_size * self.num_directions
        self.rnn = nn.LSTM(self.ipsize, self.bin_rnn_size, num_layers=args.num_layers,
                           dropout=args.dropout, bidirectional=args.bidirectional)
        self.bin_attention_1 = RecAttention(hm, args)

    def outputlength(self):
        return self.bin_rep_size

    def forward(self, single_hm, last_hidden=None):
        bin_output, hidden = self.rnn(single_hm, last_hidden)
        bin_output = bin_output.permute(1, 0, 2)  # change the dim of tensor(batch, seq_len, hidden size* num direction)
        # print('each output:', bin_output.shape)
        hm_rep, bin_alpha = self.bin_attention_1(bin_output)
        return hm_rep, bin_alpha, bin_output, hidden


class LuongAttention(torch.nn.Module):
    # Luong's attention layer
    def __init__(self, hm, args):
        super(LuongAttention, self).__init__()
        self.num_directions = 2 if args.bidirectional else 1
        if hm == False:
            self.bin_rep_size = args.bin_rnn_size * self.num_directions
        else:
            self.bin_rep_size = args.bin_rnn_size
        self.method = args.method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")

        if self.method == 'general':
            self.attn = torch.nn.Linear(self.bin_rep_size, self.bin_rep_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.bin_rep_size * 2, self.bin_rep_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(self.bin_rep_size))

    def dot_score(self, hidden, encoder_output):
        reps = batch_product_luong(hidden, encoder_output)
        return torch.sum(reps, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        reps = batch_product_luong(hidden, energy)
        return torch.sum(reps, dim=2)

    def concat_score(self, hidden, encoder_output):
        # energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        [_, batch, hidden_size] = hidden.size()
        [max_len, _, _] = encoder_output.size()
        hidden = torch.reshape(hidden, (-1, batch, hidden_size * 2)).repeat(max_len, 1, 1)
        augmentation_hidden = torch.cat((hidden, encoder_output), 2)
        augmentation_hidden = augmentation_hidden.permute(1, 0, 2)
        energy = self.attn(augmentation_hidden).tanh()
        reps = self.v * energy
        return torch.sum(reps, dim=2)

    def forward(self, hidden, encoder_outputs):
        # calculate general score, concat score and dot score individually
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        # attn_energies = attn_energies.t()  # (batch, seq_length)

        # Return the softmax normalized probability scores (with added dimension)
        alpha = F.softmax(attn_energies, dim=1).unsqueeze(1)
        reps = alpha.bmm(encoder_outputs.transpose(0, 1))
        return reps, alpha


class Recurrent_encoder2(nn.Module):
    # modular LSTM encoder
    def __init__(self, n_bins, ip_bin_size, hm, args):
        super(Recurrent_encoder2, self).__init__()
        self.bin_rnn_size = args.bin_rnn_size  # hidden size
        self.ipsize = ip_bin_size  # ipsize is input_size,  dimension of a word
        self.seq_length = n_bins
        self.num_directions = 2 if args.bidirectional else 1

        if (hm == False):
            self.bin_rnn_size = args.bin_rnn_size
        else:
            self.bin_rnn_size = args.bin_rnn_size // 2

        self.bin_rep_size = self.bin_rnn_size * self.num_directions
        self.rnn = nn.LSTM(self.ipsize, self.bin_rnn_size, num_layers=args.num_layers,
                           dropout=args.dropout, bidirectional=args.bidirectional)
        self.bin_attention_2 = LuongAttention(hm, args)

    def outputlength(self):
        return self.bin_rep_size

    def forward(self, single_hm, last_hidden=None):
        # bin_output, hidden = self.rnn(single_hm, last_hidden)  # GRU
        bin_output, (hidden, c) = self.rnn(single_hm, last_hidden) #LSTM
        # [source_length, batch_size, bin_rep_size2] = bin_output.size()
        # bin_output = bin_output.permute(1, 0, 2)  # change the dim of tensor(batch, seq_len, hidden size* num direction)
        # if encoder_outputs==None:
        #     encoder_outputs = torch.Tensor(source_length, batch_size, bin_rep_size2)
        hm_rep, bin_alpha = self.bin_attention_2(hidden, bin_output)
        return hm_rep, bin_alpha, bin_output, hidden


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Att_chrome(nn.Module):
    def __init__(self, args):
        super(Att_chrome, self).__init__()
        self.n_hms = args.n_hms
        self.n_bins = args.n_bins
        self.ip_bin_size = 1

        self.rnn_hms = nn.ModuleList()
        for i in range(self.n_hms):
            self.rnn_hms.append(Recurrent_encoder2(self.n_bins, self.ip_bin_size, False, args))
        self.opsize = self.rnn_hms[0].outputlength()
        self.hm_level_rnn_1 = Recurrent_encoder2(self.n_hms, self.opsize, True, args)
        self.opsize2 = self.hm_level_rnn_1.outputlength()
        self.diffopsize = 2 * (self.opsize2)
        self.fdiff1_1 = nn.Linear(self.opsize2, 1)

    def forward(self, iput):
        bin_attention = None
        level1_rep = None
        [batch_size, _, _] = iput.size()
        # print('opsize:', self.rnn_hms[0].outputlength())  # opsize = 64

        for hm, hm_encdr in enumerate(self.rnn_hms):
            hmod = iput[:, :, hm].contiguous()
            hmod = torch.t(hmod).unsqueeze(2)

            op, a, *_ = hm_encdr(hmod)
            # print('each op:', op.shape)
            if level1_rep is None:
                level1_rep = op
                bin_attention = a
            else:
                level1_rep = torch.cat((level1_rep, op), 1)
                bin_attention = torch.cat((bin_attention, a), 1)
        level1_rep = level1_rep.permute(1, 0, 2)
        # print('bin_attention', bin_attention.shape)
        final_rep_1, hm_level_attention_1, output, *_ = self.hm_level_rnn_1(level1_rep)
        # print('hm_level_attention_1:', hm_level_attention_1.shape)
        final_rep_1 = final_rep_1.squeeze(1)
        # print('final_rep_1 shape2:', final_rep_1.shape)
        prediction_m = ((self.fdiff1_1(final_rep_1)))

        # return torch.sigmoid(prediction_m)
        return prediction_m, hm_level_attention_1, output  # change

    # added by eric 2020.12.17
    def init_hidden(self):
        return Variable(torch.zeros(self.diffopsize, 1, self.ip_bin_size))


args_dict = {'lr': 0.0001, 'model_name': 'attchrome', 'clip': 1, 'epochs': 2, 'batch_size': 10, 'dropout': 0.8,
             'cell_1': 'Cell1', 'save_root': 'Results/Cell1', 'data_root': 'data/', 'gpuid': 0, 'gpu': 0, 'n_hms': 5,
             'n_bins': 200, 'bin_rnn_size': 32, 'num_layers': 1, 'method': 'concat', 'unidirectional': False,
             'save_attention_maps': False, 'attentionfilename': 'beta_attention.txt', 'test_on_saved_model': False,
             'bidirectional': False, 'dataset': 'Cell1', 'split': 0.01}  # lr=0.0001
att_chrome_args = AttrDict(args_dict)
att_chrome_model = Att_chrome(att_chrome_args)
