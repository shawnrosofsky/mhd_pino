
import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import FactorizedSpectralConv2d, JointFactorizedSpectralConv1d, FactorizedSpectralConv3d


class FactorizedFNO3d(nn.Module):
    def __init__(self, modes1, modes2,  modes3, width=64, fc_dim=256, n_layers=4, in_dim=4, out_dim=1,
                 pad_x=0, pad_y=0, pad_z=0,
                 non_linearity=F.gelu,
                 input_norm=None, output_norm=None,
                 joint_factorization=True, 
                 rank=1.0, 
                 factorization='cp', 
                 fixed_rank_modes=False,
                 Block=None,
                 verbose=False, 
                 fft_contraction='complex',
                 fft_norm='backward',
                 mlp=False,
                 decomposition_kwargs=dict(),
                 **kwargs):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.fc_dim = fc_dim
        self.n_layers = n_layers
        self.non_linearity = non_linearity
        self.padding = (0, 0, 0, pad_z, 0, pad_y, 0, pad_x)
        
        self.joint_factorization = joint_factorization
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.decomposition_kwargs = decomposition_kwargs
        self.fft_norm = fft_norm
        self.verbose = verbose
        
        if input_norm is not None:
            self.input_norm = torch.tensor(input_norm)
        else:
            self.input_norm = None
            
        if output_norm is not None:
            self.output_norm = torch.tensor(output_norm)
        else:
            self.output_norm = None

        if Block is None:
            Block = FactorizedSpectralConv3d
        if verbose:
            print(f'FNO Block using {Block}, fft_contraction={fft_contraction}')

        self.Block = Block

        if joint_factorization:
            self.convs = Block(self.width, self.width, self.modes1, self.modes2, self.modes3,
                               rank=rank,
                               fft_contraction=fft_contraction,
                               fft_norm=fft_norm,
                               factorization=factorization, 
                               fixed_rank_modes=fixed_rank_modes, 
                               decomposition_kwargs=decomposition_kwargs,
                               mlp=mlp,
                               n_layers=n_layers)
        else:
            self.convs = nn.ModuleList([Block(self.width, self.modes1, self.modes2, self.modes3,
                                              fft_contraction=fft_contraction,
                                              rank=rank,
                                              factorization=factorization, 
                                              fixed_rank_modes=fixed_rank_modes, 
                                              decomposition_kwargs=decomposition_kwargs,
                                              mlp=mlp,
                                              n_layers=1) for _ in range(n_layers)])
        self.linears = nn.ModuleList([nn.Conv3d(self.width, self.width, 1) for _ in range(n_layers)])
        
        self.fc0 = nn.Linear(in_dim, self.width) # input channel is 3: (a(x, y), x, y)
        self.fc1 = nn.Linear(self.width, fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)

    def forward(self, x, super_res=1):
        #grid = self.get_grid(x.shape, x.device)
        #x = torch.cat((x, grid), dim=-1)
        #x = self.fc0(x)
        #x = x.permute(0, 3, 1, 2)

        if self.input_norm is not None:
            self.input_norm = self.input_norm.to(x.device)
            x = x / self.input_norm
        
        batchsize = x.shape[0]
        nx, ny, nz = x.shape[1], x.shape[2], x.shape[3]
        x = F.pad(x, self.padding, "constant", 0)
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]
        
        # x = x.permute(0,2,3,4,1)
        x = self.fc0(x)
        x = x.permute(0,4,1,2,3)
        


        for i in range(self.n_layers):
            # if super_res > 1 and i == (self.n_layers - 1):
            #     super_res = super_res
            # else:
            #     super_res = 1

            x1 = self.convs[i](x) #, super_res=super_res)
            x2 = self.linears[i](x)
            x = x1 + x2
            if i < (self.n_layers - 1):
                x = self.non_linearity(x)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = self.non_linearity(x)
        x = self.fc2(x)
        # x = x.permute(0,4,1,2,3)
        x = x.reshape(batchsize, size_x, size_y, size_z, self.out_dim) # make sure dimensions are what we expect before getting rid of padding
        x = x[..., :nx, :ny, :nz, :]
        if self.output_norm is not None:
            self.output_norm = self.output_norm.to(x.device)
            x = x * self.output_norm
        return x


class FactorizedFNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width, fc_dim=256, n_layers=4,
                joint_factorization=True, non_linearity=F.gelu,
                rank=1.0, factorization='cp', fixed_rank_modes=False,
                domain_padding=9, in_dim=3, out_dim=1,
                Block=None,
                verbose=False, fft_contraction='complex',
                fft_norm='backward',
                decomposition_kwargs=dict()):
        super().__init__()
        """
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc_dim = fc_dim
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.domain_padding = domain_padding # pad the domain if input is non-periodic
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.decomposition_kwargs = decomposition_kwargs
        self.fft_norm = fft_norm
        self.verbose = verbose
    
        if Block is None:
            Block = FactorizedSpectralConv2d
        if verbose:
            print(f'FNO Block using {Block}, fft_contraction={fft_contraction}')

        self.Block = Block

        if joint_factorization:
            self.convs = Block(self.width, self.width, self.modes1, self.modes2, 
                               rank=rank,
                               fft_contraction=fft_contraction,
                               fft_norm=fft_norm,
                               factorization=factorization, 
                               fixed_rank_modes=fixed_rank_modes, 
                               decomposition_kwargs=decomposition_kwargs,
                               n_layers=n_layers)
        else:
            self.convs = nn.ModuleList([Block(self.width, self.width, self.modes1,
                                              self.modes2,
                                              fft_contraction=fft_contraction,
                                              rank=rank,
                                              factorization=factorization, 
                                              fixed_rank_modes=fixed_rank_modes, 
                                              decomposition_kwargs=decomposition_kwargs,
                                              n_layers=1) for _ in range(n_layers)])
        self.linears = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(n_layers)])
        
        self.fc0 = nn.Linear(in_dim, self.width) # input channel is 3: (a(x, y), x, y)
        self.fc1 = nn.Linear(self.width, fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)

    def forward(self, x, super_res=1):
        #grid = self.get_grid(x.shape, x.device)
        #x = torch.cat((x, grid), dim=-1)
        #x = self.fc0(x)
        #x = x.permute(0, 3, 1, 2)

        x = x.permute(0,2,3,1)
        x = self.fc0(x)
        x = x.permute(0,3,1,2)

        x = F.pad(x, [0, self.domain_padding, 0, self.domain_padding])

        for i in range(self.n_layers):
            if super_res > 1 and i == (self.n_layers - 1):
                super_res = super_res
            else:
                super_res = 1

            x1 = self.convs[i](x) #, super_res=super_res)
            x2 = self.linears[i](x)
            x = x1 + x2
            if i < (self.n_layers - 1):
                x = self.non_linearity(x)

        x = x[..., :-self.domain_padding, :-self.domain_padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.non_linearity(x)
        x = self.fc2(x)
        x = x.permute(0,3,1,2)
        return x

    # def extra_repr(self):
    #     s = (f'{self.modes1=}, {self.modes2=},  {self.width=}, {self.fc_dim=}, {self.n_layers=}, '
    #          f'{self.joint_factorization=}, {self.non_linearity=}, '
    #          f'{self.rank=}, {self.factorization=}, {self.fixed_rank_modes=}, '
    #          f'{self.domain_padding=}, {self.in_dim=}, {self.Block=}, '
    #          f'{self.verbose=}, '
    #          f'{self.decomposition_kwargs=}')
    #     return s

    
class FactorizedFNO1d(nn.Module):
    def __init__(self, modes, width, in_dim=2, out_dim=1, n_layers=4, 
                 lifting=None, projection=None, joint_factorization=True,  scale='auto', 
                 non_linearity=nn.GELU, rank=1.0, factorization='tucker', bias=True, 
                 fixed_rank_modes=False, fft_norm='forward', decomposition_kwargs=dict()):
        super().__init__()

        if isinstance(width, int):
            init_width = width
            final_width = width
        else:
            init_width = width[0]
            final_width = width[-1]
        
        self.non_linearity = non_linearity()

        if lifting is None:
            self.lifting = nn.Linear(in_dim, init_width)
        
        if projection is None:
            self.projection = nn.Sequential(nn.Linear(final_width, 256),
                                            self.non_linearity,
                                            nn.Linear(256, out_dim))

        self.fno_layers = JointFactorizedSpectralConv1d(modes, width, n_layers=n_layers, joint_factorization=joint_factorization,
                                                        in_dim=init_width, scale=scale, non_linearity=non_linearity,
                                                        rank=rank, factorization=factorization, bias=bias, fixed_rank_modes=fixed_rank_modes, 
                                                        fft_norm=fft_norm, decomposition_kwargs=decomposition_kwargs)
                                                        
    def forward(self, x, s=None):
        #Lifting
        x = x.permute(0,2,1)
        x = self.lifting(x)
        x = x.permute(0,2,1)

        #Fourier layers
        x = self.fno_layers(x, s=s)

        #Projection
        x = x.permute(0,2,1)
        x = self.projection(x)
        x = x.permute(0,2,1)
        
        return x
