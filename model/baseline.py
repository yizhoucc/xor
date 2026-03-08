import torch
import torch.nn as nn
import math

__all__ = ['BaselineMLP', 'BaselineCNN', 'BaselineRNN']


class BaselineMLP(nn.Module):
    """ReLU baseline MLP for comparison with XorNeuronMLP.

    Architecture: Linear → [LayerNorm] → ReLU → Dropout × N layers → Linear(num_classes)

    Set model.use_layernorm: true in config to add LayerNorm (matching XorNeuronMLP's
    normalization for fair comparison). The original paper's DenseLayerWithComplexNeurons
    always includes LayerNorm before InnerNet.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.model.input_dim
        self.out_hidden_dim = config.model.out_hidden_dim
        self.num_classes = config.model.num_classes
        self.dropout = config.model.dropout
        self.use_layernorm = getattr(config.model, 'use_layernorm', False)

        layers = []
        in_dim = self.input_dim
        for h_dim in self.out_hidden_dim:
            layers.append(nn.Linear(in_dim, h_dim))
            if self.use_layernorm:
                layers.append(nn.LayerNorm(h_dim, elementwise_affine=False))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout))
            in_dim = h_dim

        self.hidden = nn.Sequential(*layers)
        self.fc_out = nn.Linear(in_dim, self.num_classes)

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
        out = self.hidden(out)
        out = self.fc_out(out)
        loss = self.loss_func(out, labels)
        return out, loss, []


class BaselineCNN(nn.Module):
    """ReLU baseline CNN for comparison with XorNeuronConv.

    Architecture: (Conv2d → [LayerNorm] → ReLU → MaxPool2d → Dropout) × N → flatten → Linear(num_classes)

    Set model.use_layernorm: true in config to add LayerNorm (matching XorNeuronConv's
    normalization for fair comparison).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_channel = config.model.input_channel
        self.out_channel = config.model.out_channel
        self.kernel_size = config.model.kernel_size
        self.zero_pad = config.model.zero_pad
        self.stride = config.model.stride
        self.num_classes = config.model.num_classes
        self.dropout = config.model.dropout
        self.use_layernorm = getattr(config.model, 'use_layernorm', False)

        # Compute spatial dimensions through the network
        if config.dataset.name == 'mnist':
            x_size = 28
        elif config.dataset.name == 'cifar10':
            x_size = 32
        else:
            raise ValueError("Non-supported dataset!")

        conv_layers = []
        in_ch = self.input_channel
        for i in range(len(self.out_channel)):
            conv_layers.append(nn.Conv2d(in_ch, self.out_channel[i],
                                         kernel_size=self.kernel_size[i],
                                         stride=self.stride[i],
                                         padding=self.zero_pad[i]))
            x_size = (x_size - self.kernel_size[i] + 2 * self.zero_pad[i]) // self.stride[i] + 1
            if self.use_layernorm:
                conv_layers.append(nn.LayerNorm([self.out_channel[i], x_size, x_size],
                                                 elementwise_affine=False))
            conv_layers.append(nn.ReLU())
            if x_size >= 2:
                conv_layers.append(nn.MaxPool2d(kernel_size=2))
                x_size = x_size // 2
            conv_layers.append(nn.Dropout(p=self.dropout))
            in_ch = self.out_channel[i]

        self.features = nn.Sequential(*conv_layers)
        self.fc_out = nn.Linear(self.out_channel[-1] * x_size * x_size, self.num_classes)

        if self.config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Non-supported loss function!")

        self._init_param()

    def _init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias.data, -bound, bound)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, labels, collect=False):
        batch_size = x.shape[0]
        out = self.features(x)
        out = out.view(batch_size, -1)
        out = self.fc_out(out)
        loss = self.loss_func(out, labels)
        return out, loss, []


class BaselineRNN(nn.Module):
    """Tanh baseline RNN for comparison with ComplexNeuronRNN.

    Architecture: Embedding → nn.RNN(tanh) → Dropout → Linear(ntoken)
    """

    def __init__(self, config, ntoken):
        super().__init__()
        self.config = config
        self.embedding_dim = config.model.embedding_dim
        self.out_hidden_dim = config.model.out_hidden_dim
        self.ntoken = ntoken

        self.encoder = nn.Embedding(ntoken, self.embedding_dim)
        self.dropout = nn.Dropout(config.model.dropout)

        # Stack RNN layers
        self.rnn = nn.RNN(
            input_size=self.embedding_dim,
            hidden_size=self.out_hidden_dim[0],
            num_layers=len(self.out_hidden_dim),
            dropout=config.model.dropout if len(self.out_hidden_dim) > 1 else 0,
            nonlinearity='tanh',
            batch_first=False
        )
        self.decoder = nn.Linear(self.out_hidden_dim[-1], ntoken)

        if config.model.loss == 'CrossEntropy':
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
        # x: seq_len x batch_size
        # labels: seq_len * batch_size
        emb = self.dropout(self.encoder(x))

        if hx is not None:
            # Convert list of hidden states to stacked tensor for nn.RNN
            if isinstance(hx, list):
                hx = torch.stack(hx, dim=0)

        output, hidden = self.rnn(emb, hx)
        output = self.dropout(output)

        # output: seq_len x batch_size x hidden_dim → (seq_len * batch_size) x hidden_dim
        decoded = self.decoder(output.view(-1, self.out_hidden_dim[-1]))
        loss = self.loss_func(decoded, labels)

        # Convert hidden back to list for interface compatibility
        hx_list = [hidden[i] for i in range(hidden.size(0))]

        return decoded, hx_list, loss, []
