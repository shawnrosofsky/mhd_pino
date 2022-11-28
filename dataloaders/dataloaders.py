import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader, Dataset
import os
import sys 
import glob
import time
import h5py
from IPython.display import display
try:
    from .datasets import Dedalus2DDataset
except:
    from datasets import Dedalus2DDataset

class MHDDataloader(Dataset):
    def __init__(self, dataset: Dedalus2DDataset, sub_x=1, sub_t=1, ind_x=None, ind_t=None):
        self.dataset = dataset
        self.sub_x = sub_x
        self.sub_t = sub_t
        self.ind_x = ind_x
        self.ind_t = ind_t
        t, x, y = dataset.get_coords(0)
        self.x = x[:ind_x:sub_x]
        self.y = y[:ind_x:sub_x]
        self.t = t[:ind_t:sub_t]
        self.nx = len(self.x)
        self.ny = len(self.y)
        self.nt = len(self.t)
        self.num = num = len(self.dataset)
        self.x_slice = slice(0, self.ind_x, self.sub_x)
        self.t_slice = slice(0, self.ind_t, self.sub_t)
        
        # :self.ind_x:self.sub_x
        # :self.ind_t:self.sub_t
        
    def __len__(self):
        length = len(self.dataset)
        return length
    
    def __getitem__(self, index):
        fields = self.dataset[index]
        # t, x, y = self.dataset.get_coords(index)
        # nx = len(x)
        # ny = len(y)
        # nt = len(t)
        velocity = fields['velocity']
        magnetic_field = fields['magnetic field']
        # u = torch.from_numpy(velocity[:, 0])
        # v = torch.from_numpy(velocity[:, 1])
        # Bx = torch.from_numpy(magnetic_field[:, 0])
        # By = torch.from_numpy(magnetic_field[:, 1])
        u = torch.from_numpy(velocity[:self.ind_t:self.sub_t, 0, :self.ind_x:self.sub_x, :self.ind_x:self.sub_x])
        v = torch.from_numpy(velocity[:self.ind_t:self.sub_t, 1, :self.ind_x:self.sub_x, :self.ind_x:self.sub_x])
        Bx = torch.from_numpy(magnetic_field[:self.ind_t:self.sub_t, 0, :self.ind_x:self.sub_x, :self.ind_x:self.sub_x])
        By = torch.from_numpy(magnetic_field[:self.ind_t:self.sub_t, 1, :self.ind_x:self.sub_x, :self.ind_x:self.sub_x])
        # shape is now (nt, nx, ny, nfields)
        data = torch.stack([u, v, Bx, By], dim=-1)
        data0 = data[0].reshape(1, self.nx, self.ny, -1).repeat(self.nt, 1, 1, 1)
        
        grid_t = torch.from_numpy(self.t).reshape(self.nt, 1, 1, 1).repeat(1, self.nx, self.ny, 1)
        grid_x = torch.from_numpy(self.x).reshape(1, self.nx, 1, 1).repeat(self.nt, 1, self.ny, 1)
        grid_y = torch.from_numpy(self.y).reshape(1, 1, self.ny, 1).repeat(self.nt, self.nx, 1, 1)
        
        inputs = torch.cat([grid_t, grid_x, grid_y, data0], dim=-1)
        outputs = data
        
        
        # display(inputs.shape)
        # display(outputs.shape)
        
        
        return inputs, outputs
    
    def create_dataloader(self, batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
        dataloader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        return dataloader
    

class MHDDataloaderVecPot(MHDDataloader):
    def __init__(self, dataset: Dedalus2DDataset, sub_x=1, sub_t=1, ind_x=None, ind_t=None):
        super().__init__(dataset=dataset, sub_x=sub_x, sub_t=sub_t, ind_x=ind_x, ind_t=ind_t)
        
    def __getitem__(self, index):
        fields = self.dataset[index]
        # t, x, y = self.dataset.get_coords(index)
        # nx = len(x)
        # ny = len(y)
        # nt = len(t)
        velocity = fields['velocity']
        vector_potential = fields['vector potential']
        # print(velocity.shape)
        u = torch.from_numpy(velocity[:self.ind_t:self.sub_t, 0, :self.ind_x:self.sub_x, :self.ind_x:self.sub_x])
        # print(u.shape)
        v = torch.from_numpy(velocity[:self.ind_t:self.sub_t, 1, :self.ind_x:self.sub_x, :self.ind_x:self.sub_x])
        A = torch.from_numpy(vector_potential[:self.ind_t:self.sub_t, :self.ind_x:self.sub_x, :self.ind_x:self.sub_x])
        # shape is now (self.nt, self.nx, self.ny, nfields)
        data = torch.stack([u, v, A], dim=-1)
        data0 = data[0].reshape(1, self.nx, self.ny, -1).repeat(self.nt, 1, 1, 1)
        
        grid_t = torch.from_numpy(self.t).reshape(self.nt, 1, 1, 1).repeat(1, self.nx, self.ny, 1)
        grid_x = torch.from_numpy(self.x).reshape(1, self.nx, 1, 1).repeat(self.nt, 1, self.ny, 1)
        grid_y = torch.from_numpy(self.y).reshape(1, 1, self.ny, 1).repeat(self.nt, self.nx, 1, 1)
        
        inputs = torch.cat([grid_t, grid_x, grid_y, data0], dim=-1)
        outputs = data
        
        
        # display(inputs.shape)
        # display(outputs.shape)
        
        
        return inputs, outputs
    
class MHDPressureDataloader(Dataset):
    def __init__(self, dataset: Dedalus2DDataset, sub_x=1, sub_t=1):
        self.dataset = dataset
        self.sub_x = sub_x
        self.sub_t = sub_t
        t, x, y = dataset.get_coords(0)
        self.x = x[::sub_x]
        self.y = y[::sub_x]
        self.t = t[::sub_t]
        self.nx = len(self.x)
        self.ny = len(self.y)
        self.nt = len(self.t)
        self.num = num = len(self.dataset)
        
    def __len__(self):
        length = len(self.dataset)
        return length
    
    def __getitem__(self, index):
        fields = self.dataset[index]
        # t, x, y = self.dataset.get_coords(index)
        # nx = len(x)
        # ny = len(y)
        # nt = len(t)
        velocity = fields['velocity']
        magnetic_field = fields['magnetic field']
        pressure = fields['pressure']
        # u = torch.from_numpy(velocity[:, 0])
        # v = torch.from_numpy(velocity[:, 1])
        # Bx = torch.from_numpy(magnetic_field[:, 0])
        # By = torch.from_numpy(magnetic_field[:, 1])
        # p = torch.from_numpy(pressure)
        u = torch.from_numpy(velocity[:self.ind_t:self.sub_t, 0, :self.ind_x:self.sub_x, :self.ind_x:self.sub_x])
        v = torch.from_numpy(velocity[:self.ind_t:self.sub_t, 1, :self.ind_x:self.sub_x, :self.ind_x:self.sub_x])
        Bx = torch.from_numpy(magnetic_field[:self.ind_t:self.sub_t, 0, :self.ind_x:self.sub_x, :self.ind_x:self.sub_x])
        By = torch.from_numpy(magnetic_field[:self.ind_t:self.sub_t, 1, :self.ind_x:self.sub_x, :self.ind_x:self.sub_x])
        p = torch.from_numpy(pressure[:self.ind_t:self.sub_t, :self.ind_x:self.sub_x, :self.ind_x:self.sub_x])
        # shape is now (nt, nx, ny, nfields)
        data = torch.stack([u, v, Bx, By, p], dim=-1)
        data0 = data[0].reshape(1, self.nx, self.ny, -1).repeat(self.nt, 1, 1, 1)
        
        grid_t = torch.from_numpy(self.t).reshape(self.nt, 1, 1, 1).repeat(1, self.nx, self.ny, 1)
        grid_x = torch.from_numpy(self.x).reshape(1, self.nx, 1, 1).repeat(self.nt, 1, self.ny, 1)
        grid_y = torch.from_numpy(self.y).reshape(1, 1, self.ny, 1).repeat(self.nt, self.nx, 1, 1)
        
        inputs = torch.cat([grid_t, grid_x, grid_y, data0], dim=-1)
        outputs = data
        
        
        # display(inputs.shape)
        # display(outputs.shape)
        
        
        return inputs, outputs
    
    def get_norm(self):
        for inputs, outputs in self:
            inputs_std = inputs.std(dim=[0,1,2])
            outputs_std = outputs.std(dim=-1)
            inputs_std[..., :3] = 1.0
    
    def create_dataloader(self, batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
        dataloader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        return dataloader


if __name__ == "__main__":
    dataset = Dedalus2DDataset(data_path='mhd_data/simulation_outputs_mhd', output_names='output-????',field_names=['magnetic field', 'velocity', 'vector potential'])
    mhd_dataloader = MHDDataloader(dataset)
    mhd_vec_pot_dataloader = MHDDataloaderVecPot(dataset)
    
    data = mhd_dataloader[0]
    display(data)
    # dataloader[0]