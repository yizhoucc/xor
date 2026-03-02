from collections import OrderedDict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .denselayer import DenseLayerWithComplexNeurons
from .conv2dlayer import Conv2dLayerWithComplexNeurons
from .rnncell import RNNCellWithComplexNeurons

EPS = float(np.finfo(np.float32).eps)

__all__ = ['InnerNet', 'MultipleInnerNet', 'ComplexNeuronMLP', 'ComplexNeuronConv', 'ComplexNeuronRNN', 'XorNeuronMLP','XorNeuronMLP_v2','XorNeuronMLP_v3', 'XorNeuronConv',
           'XorNeuronMLP_test', 'XorNeuronConv_test']


class InnerNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.arg_in_dim = config.model.arg_in_dim

        # inner net
        if self.config.model.inner_net == 'mlp':
            self.in_hidden_dim = config.model.in_hidden_dim

            self.inner_net = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(self.arg_in_dim, self.in_hidden_dim)),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(self.in_hidden_dim, self.in_hidden_dim)),
                ('relu2', nn.ReLU()),
                ('fc3', nn.Linear(self.in_hidden_dim, 1))
            ]))
        elif self.config.model.inner_net == 'conv':
            self.in_channel = config.model.in_channel

            self.inner_net = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(self.arg_in_dim, self.in_channel, 1)),
                ('relu1', nn.ReLU()),
                ('conv2', nn.Conv2d(self.in_channel, self.in_channel, 1)),
                ('relu2', nn.ReLU()),
                ('conv3', nn.Conv2d(self.in_channel, 1, 1))
            ]))
        else:
            raise ValueError("Non-supported InnerNet!")

        self.loss_func = nn.MSELoss()
        self._init_param()

    def _init_param(self):
        mlp_modules = [
            xx for xx in [self.inner_net] if xx is not None
        ]
        for m in mlp_modules:
            if isinstance(m, nn.Sequential):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight.data)
                        if mm.bias is not None:
                            mm.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, targets):
        if self.config.model.inner_net == 'mlp':
            out = self.inner_net(x)
        elif self.config.model.inner_net == 'conv':
            sqrt_batch_size = np.int(np.sqrt(x.shape[0]))
            assert sqrt_batch_size ** 2 == x.shape[0]
            out = x.T.reshape(1, self.arg_in_dim, sqrt_batch_size, sqrt_batch_size)
            out = self.inner_net(out)
            out = out.reshape(-1, 1)

        loss = self.loss_func(out, targets)
        return out, loss


class MultipleInnerNet(nn.Module):
    def __init__(self, config):
        super(MultipleInnerNet, self).__init__()
        self.config = config
        self.num_cell_types = config.model.num_cell_types
        self.arg_in_dim = config.model.arg_in_dim

        # inner net
        if self.config.model.inner_net == 'mlp':
            self.in_hidden_dim = config.model.in_hidden_dim
            self.inner_net = nn.ModuleList()
            for i in range(self.num_cell_types):
                self.inner_net.append(
                    nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(self.arg_in_dim, self.in_hidden_dim)),
                        ('relu1', nn.ReLU()),
                        ('fc2', nn.Linear(self.in_hidden_dim, self.in_hidden_dim)),
                        ('relu2', nn.ReLU()),
                        ('fc3', nn.Linear(self.in_hidden_dim, 1))
                    ]))
                )
        elif self.config.model.inner_net == 'conv':
            self.in_channel = config.model.in_channel
            self.inner_net = nn.ModuleList()
            for i in range(self.num_cell_types):
                self.inner_net.append(
                    nn.Sequential(OrderedDict([
                        ('conv1', nn.Conv2d(self.arg_in_dim, self.in_channel, 1)),
                        ('relu1', nn.ReLU()),
                        ('conv2', nn.Conv2d(self.in_channel, self.in_channel, 1)),
                        ('relu2', nn.ReLU()),
                        ('conv3', nn.Conv2d(self.in_channel, 1, 1))
                    ]))
                )
        else:
            raise ValueError("Non-supported InnerNet!")


class ComplexNeuronMLP(nn.Module):

    def __init__(self, config):
        super(ComplexNeuronMLP, self).__init__()
        self.config = config
        self.num_cell_types = config.model.num_cell_types
        self.input_dim = config.model.input_dim
        self.arg_in_dim = config.model.arg_in_dim
        self.in_hidden_dim = config.model.in_hidden_dim
        self.out_hidden_dim = config.model.out_hidden_dim
        self.num_classes = config.model.num_classes
        self.dropout = config.model.dropout

        # inner net
        self.inner_net = nn.ModuleList()
        for i in range(self.num_cell_types):
            self.inner_net.append(
                nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(self.arg_in_dim, self.in_hidden_dim)),
                    ('relu1', nn.ReLU()),
                    ('fc2', nn.Linear(self.in_hidden_dim, self.in_hidden_dim)),
                    ('relu2', nn.ReLU()),
                    ('fc3', nn.Linear(self.in_hidden_dim, 1))
                ]))
            )

        # outer net
        self.outer_net = nn.ModuleList()
        for i in range(len(self.out_hidden_dim)):
            if i == 0:
                self.outer_net.append(
                    DenseLayerWithComplexNeurons(self.inner_net,
                                                 self.arg_in_dim,
                                                 self.input_dim,
                                                 self.out_hidden_dim[i])
                )
            else:
                self.outer_net.append(
                    DenseLayerWithComplexNeurons(self.inner_net,
                                                 self.arg_in_dim,
                                                 self.out_hidden_dim[i - 1],
                                                 self.out_hidden_dim[i])
                )

        # output layer
        self.fc_out = nn.Linear(self.out_hidden_dim[-1], self.num_classes)
        # dropout layer
        self.drop_layer = nn.Dropout(p=self.dropout)

        if self.config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Non-supported loss function!")

        # self._init_param()

    def _init_param(self):
        mlp_modules = [
            xx for xx in [self.fc_out, self.inner_net] if xx is not None
        ]
        for m in mlp_modules:
            if isinstance(m, nn.Sequential):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight.data)
                        if mm.bias is not None:
                            mm.bias.data.zero_()
            elif isinstance(m, nn.ModuleList):
                for mm in m:
                    if isinstance(mm, nn.Sequential):
                        for mmm in mm:
                            if isinstance(mmm, nn.Linear):
                                nn.init.xavier_uniform_(mmm.weight.data)
                                if mmm.bias is not None:
                                    mmm.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, labels, collect=False):
        x = x.reshape(-1, np.array(x.shape[1:]).prod())
        # in2cells_per_layer : batch_size x num_layers x num_cell_types x ... x arity
        in2cells_per_layer = []
        for i, fc in enumerate(self.outer_net):
            out, in2cells = fc(out, collect=collect) if i > 0 else fc(x, collect=collect)
            in2cells_per_layer.append(in2cells)
            out = self.drop_layer(out)

        out = self.fc_out(out)
        loss = self.loss_func(out, labels)

        return out, loss, in2cells_per_layer


class ComplexNeuronConv(nn.Module):

    def __init__(self, config):
        super(ComplexNeuronConv, self).__init__()
        self.config = config
        self.num_cell_types = config.model.num_cell_types
        self.input_channel = config.model.input_channel
        self.arg_in_dim = config.model.arg_in_dim
        self.in_channel = config.model.in_channel
        self.out_channel = config.model.out_channel
        self.kernel_size = config.model.kernel_size
        self.zero_pad = config.model.zero_pad
        self.stride = config.model.stride
        self.num_classes = config.model.num_classes
        self.dropout = config.model.dropout

        # inner net
        self.inner_net = nn.ModuleList()
        for i in range(self.num_cell_types):
            self.inner_net.append(
                nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(self.arg_in_dim, self.in_channel, 1)),
                    ('relu1', nn.ReLU()),
                    ('conv2', nn.Conv2d(self.in_channel, self.in_channel, 1)),
                    ('relu2', nn.ReLU()),
                    ('conv3', nn.Conv2d(self.in_channel, 1, 1))
                ]))
            )
        # max-pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        # outer net
        if self.config.dataset.name == 'mnist':
            input_shape = [1, 1, 28, 28]
        elif self.config.dataset.name == 'cifar10':
            input_shape = [1, 3, 32, 32]
        else:
            raise ValueError("Non-supported dataset!")

        self.outer_net = nn.ModuleList()
        for i in range(len(self.out_channel)):
            if i == 0:
                self.outer_net.append(
                    Conv2dLayerWithComplexNeurons(self.inner_net,
                                                  self.arg_in_dim,
                                                  self.input_channel,
                                                  self.out_channel[i],
                                                  self.kernel_size[i],
                                                  stride=self.stride[i],
                                                  padding=self.zero_pad[i])
                )
            elif i < len(self.out_channel) - 1:
                self.outer_net.append(
                    Conv2dLayerWithComplexNeurons(self.inner_net,
                                                  self.arg_in_dim,
                                                  self.out_channel[i - 1],
                                                  self.out_channel[i],
                                                  self.kernel_size[i],
                                                  stride=self.stride[i],
                                                  padding=self.zero_pad[i])
                )
            else:
                # calculate the expected shape of input to fc_out_1
                for j in range(len(self.out_channel) - 1):
                    input_shape = self.outer_net[j](torch.rand(*input_shape))[0].data.shape
                    input_shape = self.max_pool(torch.rand(*input_shape)).data.shape

                self.outer_net.append(
                    Conv2dLayerWithComplexNeurons(self.inner_net,
                                                  self.arg_in_dim,
                                                  np.prod(list(input_shape)),
                                                  self.out_channel[i],
                                                  self.kernel_size[i],
                                                  stride=self.stride[i],
                                                  padding=self.zero_pad[i])
                )

        # output layer
        self.fc_out = nn.Linear(self.out_channel[-1], self.num_classes)
        # dropout layer
        self.drop_layer = nn.Dropout(p=self.dropout)

        if self.config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Non-supported loss function!")

        # self._init_param()

    def _init_param(self):
        mlp_modules = [
            xx for xx in [self.fc_out, self.inner_net] if xx is not None
        ]
        for m in mlp_modules:
            if isinstance(m, nn.Sequential):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight.data)
                        if mm.bias is not None:
                            mm.bias.data.zero_()
            elif isinstance(m, nn.ModuleList):
                for mm in m:
                    if isinstance(mm, nn.Sequential):
                        for mmm in mm:
                            if isinstance(mmm, nn.Conv2d):
                                nn.init.kaiming_uniform_(mmm.weight.data, a=math.sqrt(5))
                                if mmm.bias is not None:
                                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                                    bound = 1 / math.sqrt(fan_in)
                                    nn.init.uniform_(mmm.bias.data, -bound, bound)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, labels, collect=False):
        batch_size = x.shape[0]
        # in2cells_per_layer : num_layers x [batch_size x num_cell_types x ... x arity]
        in2cells_per_layer = []
        for i, conv in enumerate(self.outer_net[:-1]):
            out, in2cells = conv(out, collect=collect) if i > 0 else conv(x, collect=collect)
            if i == 2: # collect 2nd layer only
                in2cells_per_layer.append(in2cells)
            out = self.max_pool(out)
            out = self.drop_layer(out)

        # Effective flattening via 1x1 conv
        out, in2cells = self.outer_net[-1](out.view(batch_size, -1, 1, 1), collect=collect)
        in2cells_per_layer.append(in2cells)
        out = self.drop_layer(out)
        # Output Layer
        out = self.fc_out(out.view(batch_size, -1))
        loss = self.loss_func(out, labels)

        return out, loss, in2cells_per_layer


class ComplexNeuronRNN(nn.Module):

    def __init__(self, config, ntoken):
        super(ComplexNeuronRNN, self).__init__()
        self.config = config
        self.num_cell_types = config.model.num_cell_types
        self.embedding_dim = config.model.embedding_dim
        self.arg_in_dim = config.model.arg_in_dim
        self.in_hidden_dim = config.model.in_hidden_dim
        self.out_hidden_dim = config.model.out_hidden_dim

        self.dropout = nn.Dropout(config.model.dropout)
        self.encoder = nn.Embedding(ntoken, self.embedding_dim)  # Token2Embeddings
        # inner net
        self.inner_net = nn.ModuleList()
        for i in range(self.num_cell_types):
            self.inner_net.append(
                nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(self.arg_in_dim, self.in_hidden_dim)),
                    ('relu1', nn.ReLU()),
                    ('fc2', nn.Linear(self.in_hidden_dim, self.in_hidden_dim)),
                    ('relu2', nn.ReLU()),
                    ('fc3', nn.Linear(self.in_hidden_dim, 1))
                ]))
            )

        # outer net
        self.outer_net = nn.ModuleList()
        for i in range(len(self.out_hidden_dim)):
            if i == 0:
                self.outer_net.append(
                    RNNCellWithComplexNeurons(self.inner_net,
                                              self.arg_in_dim,
                                              self.embedding_dim,
                                              self.out_hidden_dim[i])
                )
            else:
                self.outer_net.append(
                    RNNCellWithComplexNeurons(self.inner_net,
                                              self.arg_in_dim,
                                              self.out_hidden_dim[i - 1],
                                              self.out_hidden_dim[i])
                )
        self.decoder = nn.Linear(self.out_hidden_dim[-1], ntoken)

        if self.config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Non-supported loss function!")

        self._init_param()

    def _init_param(self):
        initrange = 0.05
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, labels, mask=None, hx=None):
        # input : seq_len, batch_size
        # labels : seq_len * batch_size
        # emb : seq_len, batch_size, emb_dim
        # hx : num_layer * [batch_size, hidden_dim]
        emb = self.dropout(self.encoder(x))
        if not hx:
            hx = [torch.zeros(x.size(1), self.out_hidden_dim[i], device=x.device) for i in range(len(self.out_hidden_dim))]
        if not mask:
            mask = []
            for i in range(len(self.out_hidden_dim)):
                mask.append(torch.ones(self.out_hidden_dim[i], device=x.device))

        # in2cells_per_layer : batch_size x num_layers x num_cell_types x ... x arity
        in2cells_per_layer = []
        output = []
        for i_seq in range(emb.size(0)):
            for i_layer, rnn_layer in enumerate(self.outer_net):
                if i_layer == 0 and i_seq == 0:
                    hx_update, in2cells = rnn_layer(emb[i_seq])
                elif i_layer > 0 and i_seq == 0:
                    hx_update, in2cells = rnn_layer(hx[i_layer-1])
                elif i_layer == 0 and i_seq > 0:
                    hx_update, in2cells = rnn_layer(emb[i_seq], hx[i_layer])
                else:
                    hx_update, in2cells = rnn_layer(hx[i_layer-1], hx[i_layer])

                hx[i_layer] = hx_update * mask[i_layer]
                if i_seq == emb.size(0) - 1:
                    in2cells_per_layer.append(in2cells)
            out = self.decoder(hx[-1])
            output.append(out)

        # output : seq_len * batch_size, ntoken
        output = torch.cat(output, 0)
        # labels : seq_len * batch_size
        loss = self.loss_func(output, labels)

        return output, hx, loss, in2cells_per_layer


class XorNeuronMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.model.input_dim
        self.arg_in_dim = config.model.arg_in_dim
        self.in_hidden_dim = config.model.in_hidden_dim
        self.out_hidden_dim = config.model.out_hidden_dim
        self.num_classes = config.model.num_classes
        self.dropout = config.model.dropout

        # inner net
        self.inner_net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.arg_in_dim, self.in_hidden_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.in_hidden_dim, self.in_hidden_dim)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(self.in_hidden_dim, 1))
        ]))

        # outer net
        self.outer_net = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        for i in range(len(self.out_hidden_dim)):
            self.layer_norm.append(nn.LayerNorm(self.out_hidden_dim[i], elementwise_affine=False))
            if i == 0:
                self.outer_net.append(nn.Linear(self.input_dim, self.out_hidden_dim[i]))
            else:
                self.outer_net.append(nn.Linear(self.out_hidden_dim[i] // self.arg_in_dim, self.out_hidden_dim[i]))

        # output layer
        self.fc_out = nn.Linear(self.out_hidden_dim[-1] // self.arg_in_dim, self.num_classes)
        # dropout layer
        self.drop_layer = nn.Dropout(p=self.dropout)

        if self.config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Non-supported loss function!")

        self._init_param()

    def _init_param(self):
        mlp_modules = [
            xx for xx in [self.fc_out, self.inner_net, self.outer_net] if xx is not None
        ]
        for m in mlp_modules:
            if isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight.data)
                        if mm.bias is not None:
                            mm.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    # def forward(self, x, labels):
    #     batch_size = x.shape[0]
    #     x = x.reshape(-1, np.array(x.shape[1:]).prod())

    #     for i, fc in enumerate(self.outer_net):
    #         out = fc(out) if i > 0 else fc(x)
    #         out = self.layer_norm[i](out)
    #         out = self.inner_net(out.reshape(batch_size, -1, self.arg_in_dim)).reshape(batch_size, -1)
    #         out = self.drop_layer(out)

    #     out = self.fc_out(out)
    #     loss = self.loss_func(out, labels)

    #     return out, loss
    def forward(self, x, labels, collect=False): # 🚀 增加 collect 参数适配 Runner
            batch_size = x.shape[0]
            # 确保输入展平
            out = x.reshape(batch_size, -1)
            in2cells_per_layer = [] # 🚀 创建占位列表

            for i, fc in enumerate(self.outer_net):
                out = fc(out) 
                out = self.layer_norm[i](out)
                
                # 🚀 记录中间层输入（对 Phase 2 的可视化非常重要）
                reshaped_out = out.reshape(batch_size, -1, self.arg_in_dim)
                if collect:
                    in2cells_per_layer.append(reshaped_out.data.cpu().numpy())

                # 执行 InnerNet 激活
                # .reshape(batch_size, -1) 将 [B, Neurons, 1] 转回 [B, Neurons]
                out = self.inner_net(reshaped_out).reshape(batch_size, -1)
                out = self.drop_layer(out)

            out = self.fc_out(out)
            loss = self.loss_func(out, labels)

            # 🚀 关键：返回 3 个值以匹配 Runner 的解包逻辑
            return out, loss, in2cells_per_layer

class XorNeuronConv(nn.Module):

    def __init__(self, config):
        super(XorNeuronConv, self).__init__()
        self.config = config
        self.input_channel = config.model.input_channel
        self.arg_in_dim = config.model.arg_in_dim
        self.in_channel = config.model.in_channel
        self.out_channel = config.model.out_channel
        self.kernel_size = config.model.kernel_size
        self.zero_pad = config.model.zero_pad
        self.stride = config.model.stride
        self.num_classes = config.model.num_classes
        self.dropout = config.model.dropout

        # inner net
        self.inner_net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.arg_in_dim, self.in_channel, 1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(self.in_channel, self.in_channel, 1)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(self.in_channel, 1, 1))
        ]))

        # outer net
        if self.config.dataset.name == 'mnist':
            x_shape = [28]
        elif self.config.dataset.name == 'cifar10':
            x_shape = [32]
        else:
            raise ValueError("Non-supported dataset!")

        self.outer_net = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        for i in range(len(self.out_channel)):
            x_shape.append(((x_shape[-1] - self.kernel_size[i] + 2 * self.zero_pad[i]) // self.stride[i] + 1))
            self.layer_norm.append(
                nn.LayerNorm([self.out_channel[i], x_shape[-1], x_shape[-1]], elementwise_affine=False))
            if x_shape[-1] >= 2:
                x_shape.append(x_shape[-1] // 2)
            else:
                x_shape.append(x_shape[-1])  # no pooling
            if i == 0:
                self.outer_net.append(nn.Conv2d(in_channels=self.input_channel,
                                                out_channels=self.out_channel[i],
                                                kernel_size=self.kernel_size[i],
                                                stride=self.stride[i],
                                                padding=self.zero_pad[i]))
            else:
                self.outer_net.append(nn.Conv2d(in_channels=self.out_channel[i - 1] // self.arg_in_dim,
                                                out_channels=self.out_channel[i],
                                                kernel_size=self.kernel_size[i],
                                                stride=self.stride[i],
                                                padding=self.zero_pad[i]))

        # max-pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        # output layer
        self.fc_out = nn.ModuleList()
        # self.fc_out.append(nn.Linear(self.out_channel[-1] // self.arg_in_dim + x_shape ** 2, 256))
        self.fc_out.append(nn.Conv2d(in_channels=self.out_channel[-1] // self.arg_in_dim * x_shape[-1] * x_shape[-1],
                                     out_channels=256,
                                     kernel_size=1,
                                     stride=1))
        self.fc_out.append(nn.Linear(256 // self.arg_in_dim, self.num_classes))

        # dropout layer
        self.drop_layer = nn.Dropout(p=self.dropout)

        if self.config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Non-supported loss function!")

        self._init_param()

    def _init_param(self):
        mlp_modules = [
            xx for xx in [self.fc_out, self.inner_net, self.outer_net] if xx is not None
        ]
        for m in mlp_modules:
            if isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight.data)
                        if mm.bias is not None:
                            mm.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, labels, collect=False):
        batch_size = x.shape[0]
        in2cells_per_layer = []

        for i, conv in enumerate(self.outer_net):
            # ConvLayer
            out = conv(out) if i > 0 else conv(x)
            out = self.layer_norm[i](out)
            # InnerNet
            out = out.reshape(-1, self.arg_in_dim, out.shape[2], out.shape[3])
            if collect:
                in2cells_per_layer.append(
                    np.moveaxis(out.data.cpu().numpy(), 1, -1).reshape(-1, self.arg_in_dim))
            out = self.inner_net(out)
            out = out.reshape(batch_size, out.shape[0] // batch_size, out.shape[-2], out.shape[-1])
            # MaxPooling (skip if spatial too small)
            if out.shape[-1] >= 2:
                out = self.max_pool(out)
            out = self.drop_layer(out)

        # Effective flattening via 1x1 conv
        out = self.fc_out[0](out.view(batch_size, -1, 1, 1))
        # InnerNet
        out = out.reshape(-1, self.arg_in_dim, out.shape[2], out.shape[3])
        if collect:
            in2cells_per_layer.append(
                np.moveaxis(out.data.cpu().numpy(), 1, -1).reshape(-1, self.arg_in_dim))
        out = self.inner_net(out)
        out = out.reshape(batch_size, out.shape[0] // batch_size, out.shape[-2], out.shape[-1])
        # OutputLayer
        out = self.fc_out[1](out.view(batch_size, -1))
        loss = self.loss_func(out, labels)

        return out, loss, in2cells_per_layer


class XorNeuronMLP_test(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.model.input_dim
        self.arg_in_dim = config.model.arg_in_dim
        self.in_hidden_dim = config.model.in_hidden_dim
        self.out_hidden_dim = config.model.out_hidden_dim
        self.num_classes = config.model.num_classes
        self.dropout = config.model.dropout

        # inner net
        self.inner_net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.arg_in_dim, self.in_hidden_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.in_hidden_dim, self.in_hidden_dim)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(self.in_hidden_dim, 1))
        ]))

        # outer net
        self.outer_net = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        for i in range(len(self.out_hidden_dim)):
            self.layer_norm.append(nn.LayerNorm(self.out_hidden_dim[i], elementwise_affine=False))
            if i == 0:
                self.outer_net.append(nn.Linear(self.input_dim, self.out_hidden_dim[i]))
            else:
                self.outer_net.append(nn.Linear(self.out_hidden_dim[i] // self.arg_in_dim, self.out_hidden_dim[i]))

        # output layer
        self.fc_out = nn.Linear(self.out_hidden_dim[-1] // self.arg_in_dim, self.num_classes)
        # dropout layer
        self.drop_layer = nn.Dropout(p=self.dropout)

        if self.config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Non-supported loss function!")

        self._init_param()

    def _init_param(self):
        mlp_modules = [
            xx for xx in [self.fc_out, self.inner_net, self.outer_net] if xx is not None
        ]
        for m in mlp_modules:
            if isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight.data)
                        if mm.bias is not None:
                            mm.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, labels):
        batch_size = x.shape[0]
        x = x.reshape(-1, np.array(x.shape[1:]).prod())
        input2inner = {}

        for i, fc in enumerate(self.outer_net):
            out = fc(out) if i > 0 else fc(x)
            out = self.layer_norm[i](out)
            # Collect Inputs to InnerNet
            input2inner[i] = out.data.cpu().numpy()
            out = self.inner_net(out.reshape(batch_size, -1, self.arg_in_dim)).reshape(batch_size, -1)
            out = self.drop_layer(out)

        out = self.fc_out(out)
        loss = self.loss_func(out, labels)

        return out, loss, input2inner


class XorNeuronConv_test(nn.Module):

    def __init__(self, config):
        super(XorNeuronConv_test, self).__init__()
        self.config = config
        self.input_channel = config.model.input_channel
        self.arg_in_dim = config.model.arg_in_dim
        self.in_channel = config.model.in_channel
        self.out_channel = config.model.out_channel
        self.kernel_size = config.model.kernel_size
        self.zero_pad = config.model.zero_pad
        self.stride = config.model.stride
        self.num_classes = config.model.num_classes
        self.dropout = config.model.dropout

        # inner net
        self.inner_net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.arg_in_dim, self.in_channel, 1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(self.in_channel, self.in_channel, 1)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(self.in_channel, 1, 1))
        ]))

        # outer net
        if self.config.dataset.name == 'mnist':
            x_shape = [28]
        elif self.config.dataset.name == 'cifar10':
            x_shape = [32]
        else:
            raise ValueError("Non-supported dataset!")

        self.outer_net = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        for i in range(len(self.out_channel)):
            x_shape.append(((x_shape[-1] - self.kernel_size[i] + 2 * self.zero_pad[i]) // self.stride[i] + 1))
            self.layer_norm.append(
                nn.LayerNorm([self.out_channel[i], x_shape[-1], x_shape[-1]], elementwise_affine=False))
            if x_shape[-1] >= 2:
                x_shape.append(x_shape[-1] // 2)
            else:
                x_shape.append(x_shape[-1])  # no pooling
            if i == 0:
                self.outer_net.append(nn.Conv2d(in_channels=self.input_channel,
                                                out_channels=self.out_channel[i],
                                                kernel_size=self.kernel_size[i],
                                                stride=self.stride[i],
                                                padding=self.zero_pad[i]))
            else:
                self.outer_net.append(nn.Conv2d(in_channels=self.out_channel[i - 1] // self.arg_in_dim,
                                                out_channels=self.out_channel[i],
                                                kernel_size=self.kernel_size[i],
                                                stride=self.stride[i],
                                                padding=self.zero_pad[i]))

        # max-pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        # output layer
        self.fc_out = nn.ModuleList()
        # self.fc_out.append(nn.Linear(self.out_channel[-1] // self.arg_in_dim + x_shape ** 2, 256))
        self.fc_out.append(nn.Conv2d(in_channels=self.out_channel[-1] // self.arg_in_dim * x_shape[-1] * x_shape[-1],
                                     out_channels=256,
                                     kernel_size=1,
                                     stride=1))
        self.fc_out.append(nn.Linear(256 // self.arg_in_dim, self.num_classes))

        # dropout layer
        self.drop_layer = nn.Dropout(p=self.dropout)

        if self.config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Non-supported loss function!")

        self._init_param()

    def _init_param(self):
        mlp_modules = [
            xx for xx in [self.fc_out, self.inner_net, self.outer_net] if xx is not None
        ]
        for m in mlp_modules:
            if isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight.data)
                        if mm.bias is not None:
                            mm.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, labels):
        batch_size = x.shape[0]
        input2inner = {}
        for i, conv in enumerate(self.outer_net):
            # ConvLayer
            out = conv(out) if i > 0 else conv(x)
            out = self.layer_norm[i](out)
            out = out.reshape(-1, self.arg_in_dim, out.shape[2], out.shape[3])
            # Collect Inputs to InnerNet
            input2inner[i] = np.moveaxis(out.data.cpu().numpy(), 1, -1).reshape(-1, self.arg_in_dim)
            # InnerNet
            out = self.inner_net(out)
            out = out.reshape(batch_size, out.shape[0] // batch_size, out.shape[-2], out.shape[-1])
            # MaxPooling (skip if spatial too small)
            if out.shape[-1] >= 2:
                out = self.max_pool(out)
            out = self.drop_layer(out)

        # Effective flattening via 1x1 conv
        out = self.fc_out[0](out.view(batch_size, -1, 1, 1))
        # InnerNet
        out = out.reshape(-1, self.arg_in_dim, out.shape[2], out.shape[3])
        # Collect Inputs to InnerNet
        input2inner[i + 1] = np.moveaxis(out.data.cpu().numpy(), 1, -1).reshape(-1, self.arg_in_dim)
        out = self.inner_net(out)
        out = out.reshape(batch_size, out.shape[0] // batch_size, out.shape[-2], out.shape[-1])
        # OutputLayer
        out = self.fc_out[1](out.view(batch_size, -1))
        loss = self.loss_func(out, labels)

        return out, loss, input2inner


class XorNeuronMLP_v3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_cell_types = config.model.num_cell_types
        self.input_dim = config.model.input_dim
        self.arg_in_dim = config.model.arg_in_dim
        self.in_hidden_dim = config.model.in_hidden_dim
        self.out_hidden_dim = config.model.out_hidden_dim
        self.num_classes = config.model.num_classes
        self.dropout = config.model.dropout

        # 🚀 [V3 CORE] Merge multiple InnerNets into single Grouped Layers
        # Instead of a ModuleList of Sequential nets, we make 3 Grouped Conv1d layers.
        # This acts as a parallel 'Batch' of activation functions.
        self.inner_net = nn.ModuleList()
        for i in range(3): # Standard InnerNet has 3 layers: fc1, fc2, fc3
            in_d = self.arg_in_dim if i == 0 else self.in_hidden_dim
            out_d = 1 if i == 2 else self.in_hidden_dim
            
            # Conv1d with groups=num_cell_types creates independent 'routes'
            # for each neuron type, running them all in one GPU kernel launch.
            self.inner_net.append(nn.Conv1d(
                in_channels=in_d * self.num_cell_types,
                out_channels=out_d * self.num_cell_types,
                kernel_size=1,
                groups=self.num_cell_types
            ))

        # Outer Network (Standard Linear Layers)
        self.outer_net = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        for i in range(len(self.out_hidden_dim)):
            self.layer_norm.append(nn.LayerNorm(self.out_hidden_dim[i], elementwise_affine=False))
            if i == 0:
                self.outer_net.append(nn.Linear(self.input_dim, self.out_hidden_dim[i]))
            else:
                prev_dim = self.out_hidden_dim[i-1] // self.arg_in_dim
                self.outer_net.append(nn.Linear(prev_dim, self.out_hidden_dim[i]))

        self.fc_out = nn.Linear(self.out_hidden_dim[-1] // self.arg_in_dim, self.num_classes)
        self.drop_layer = nn.Dropout(p=self.dropout)
        self.loss_func = nn.CrossEntropyLoss()
        self._init_param()

    def _init_param(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, labels, collect=False):
        batch_size = x.shape[0]
        out = x.reshape(batch_size, -1)
        in2cells_per_layer = [] 

        for i, fc in enumerate(self.outer_net):
            out = fc(out) 
            out = self.layer_norm[i](out)
            
            # 🚀 [V3 LOGIC] Reshape to route features to the right 'Group'
            # Current shape: [Batch, Total_Neurons * Arity]
            num_neurons_total = out.shape[1] // self.arg_in_dim
            neurons_per_type = num_neurons_total // self.num_cell_types
            
            # Step A: [Batch, Num_Types, Neurons_per_Type, Arity]
            out = out.reshape(batch_size, self.num_cell_types, neurons_per_type, self.arg_in_dim)
            
            if collect:
                in2cells_per_layer.append(out.data.cpu().numpy())
            
            # Step B: Permute to [Batch, Num_Types * Arity, Neurons_per_Type]
            # This prepares the data for the Grouped Conv1d engine.
            out = out.permute(0, 1, 3, 2).reshape(batch_size, self.num_cell_types * self.arg_in_dim, neurons_per_type)
            
            # Step C: One GPU Launch for ALL cell types
            out = torch.relu(self.inner_net[0](out))
            out = torch.relu(self.inner_net[1](out))
            out = self.inner_net[2](out) # Result: [Batch, Num_Types * 1, Neurons_per_Type]
            
            # Step D: Flatten back to [Batch, Total_Neurons]
            out = out.reshape(batch_size, -1)
            out = self.drop_layer(out)

        out = self.fc_out(out)
        loss = self.loss_func(out, labels)
        
        return out, loss, in2cells_per_layer
    
    
class XorNeuronMLP_v2(nn.Module):
    def __init__(self, config):
        super(XorNeuronMLP_v2, self).__init__()
        self.config = config
        self.num_cell_types = config.model.num_cell_types
        self.input_dim = config.model.input_dim
        self.arg_in_dim = config.model.arg_in_dim
        self.in_hidden_dim = config.model.in_hidden_dim
        self.out_hidden_dim = config.model.out_hidden_dim
        self.num_classes = config.model.num_classes
        self.dropout = config.model.dropout

        # 🚀 inner_net 必须是 nn.ModuleList 以支持多类型加载
        self.inner_net = nn.ModuleList()
        for i in range(self.num_cell_types):
            self.inner_net.append(nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(self.arg_in_dim, self.in_hidden_dim)),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(self.in_hidden_dim, self.in_hidden_dim)),
                ('relu2', nn.ReLU()),
                ('fc3', nn.Linear(self.in_hidden_dim, 1))
            ])))

        # 外部网络与归一化层
        self.outer_net = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        for i in range(len(self.out_hidden_dim)):
            self.layer_norm.append(nn.LayerNorm(self.out_hidden_dim[i], elementwise_affine=False))
            if i == 0:
                self.outer_net.append(nn.Linear(self.input_dim, self.out_hidden_dim[i]))
            else:
                prev_dim = self.out_hidden_dim[i-1] // self.arg_in_dim
                self.outer_net.append(nn.Linear(prev_dim, self.out_hidden_dim[i]))

        self.fc_out = nn.Linear(self.out_hidden_dim[-1] // self.arg_in_dim, self.num_classes)
        self.drop_layer = nn.Dropout(p=self.dropout)
        
        if self.config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Non-supported loss function!")

        self._init_param()

    def _init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, labels, collect=False):
        batch_size = x.shape[0]
        out = x.reshape(batch_size, -1)
        in2cells_per_layer = [] # 占位符

        for i, fc in enumerate(self.outer_net):
            out = fc(out) 
            out = self.layer_norm[i](out)
            
            # 🚀 向量化分组处理逻辑
            reshaped_out = out.reshape(batch_size, -1, self.arg_in_dim)
            
            # 记录输入数据 (可选，用于可视化)
            if collect:
                in2cells_per_layer.append(reshaped_out.data.cpu().numpy())
            
            # 按类型切分并并行激活
            cell_groups = torch.chunk(reshaped_out, self.num_cell_types, dim=1)
            group_outputs = []
            for j, group in enumerate(cell_groups):
                activated = self.inner_net[j](group) 
                group_outputs.append(activated.reshape(batch_size, -1))
            
            out = torch.cat(group_outputs, dim=1)
            out = self.drop_layer(out)

        out = self.fc_out(out)
        loss = self.loss_func(out, labels)
        
        # 🚀 修复返回值：必须返回 3 个值以匹配 Runner
        return out, loss, in2cells_per_layer

class TEST:
    def __init__(self):
        pass