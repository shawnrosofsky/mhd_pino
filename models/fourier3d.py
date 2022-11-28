import torch
import torch.nn as nn
import torch.nn.functional as F

from .basics import SpectralConv3d


class FNN3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width=16, fc_dim=128, layers=None, in_dim=4, out_dim=1, activation='tanh', pad_x=0, pad_y=0, pad_z=0, input_norm=None, output_norm=None):
        '''
        Args:
            modes1: list of int, first dimension maximal modes for each layer
            modes2: list of int, second dimension maximal modes for each layer
            modes3: list of int, third dimension maximal modes for each layer
            layers: list of int, channels for each layer
            in_dim: int, input dimension
            out_dim: int, output dimension
        '''
        super(FNN3d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.padding = (0, 0, 0, pad_z, 0, pad_y, 0, pad_x)
       
        if input_norm is not None:
            self.input_norm = torch.tensor(input_norm)
        else:
            self.input_norm = None
            
        if output_norm is not None:
            self.output_norm = torch.tensor(output_norm)
        else:
            self.output_norm = None

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(self.in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv3d(
            in_size, out_size, mode1_num, mode2_num, mode3_num)
            for in_size, out_size, mode1_num, mode2_num, mode3_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2, self.modes3)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, self.out_dim)
        
        if activation =='tanh':
            self.activation = F.tanh
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation == F.relu
        elif activation == 'linear':
            self.activation == F.linear
        elif activation == 'bilinear':
            self.activation == F.bilinear
        elif activation == 'hardtanh':
            self.activation == F.hardtanh
        elif activation == 'relu6':
            self.activation == F.relu6
        elif activation == 'elu':
            self.activation == F.elu
        elif activation == 'selu':
            self.activation == F.selu
        elif activation == 'celu':
            self.activation == F.celu
        elif activation == 'leaky_relu':
            self.activation == F.leaky_relu
        elif activation == 'prelu':
            self.activation == F.prelu
        elif activation == 'rrelu':
            self.activation == F.rrelu
        elif activation == 'silu':
            self.activation == F.silu
        elif activation == 'mish':
            self.activation == F.mish
        
        else:
            self.activation = activation
            # raise ValueError(f'{activation} is not supported')

    def forward(self, x):
        '''
        Args:
            x: (batchsize, t_grid, x_grid, y_grid, in_dim=3+fields_in)
        Returns:
            u: (batchsize, t_grid, x_grid, y_grid, out_dim=fields_out)
        '''
        length = len(self.ws)
        if self.input_norm is not None:
            self.input_norm = self.input_norm.to(x.device)
            x = x / self.input_norm
        batchsize = x.shape[0]
        nx, ny, nz = x.shape[1], x.shape[2], x.shape[3]
        x = F.pad(x, self.padding, "constant", 0)
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y, size_z)
            x = x1 + x2
            if i != length - 1:
                # x = torch.tanh(x)
                x = self.activation(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        # x = torch.tanh(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.reshape(batchsize, size_x, size_y, size_z, self.out_dim) # make sure dimensions are what we expect before getting rid of padding
        x = x[..., :nx, :ny, :nz, :]
        if self.output_norm is not None:
            self.output_norm = self.output_norm.to(x.device)
            x = x * self.output_norm
        return x