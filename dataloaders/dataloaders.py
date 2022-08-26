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
from datasets import Dedalus2DDataset

class MHDDataloader(Dataset):
    def __init__(self, dataset: Dedalus2DDataset, sub_x=1, sub_t=1, num_train=None):
        # field_names=['magnetic field', 'velocity', 'pressure']
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
        if (num_train is None) or (num_train > len(self.dataset)):
            num_train = len(self.dataset)
        self.num_train = num_train
        self.num_test = num - num_train
        
    def __len__(self):
        length = len(self.dataset)
        return length
    
    def __getitem__(self, index):
        fields = self.dataset[index]
        t, x, y = self.dataset.get_coords(index)
        nx = len(x)
        ny = len(y)
        nt = len(t)
        velocity = fields['velocity']
        magnetic_field = fields['magnetic field']
        u = torch.from_numpy(velocity[:, 0])
        v = torch.from_numpy(velocity[:, 1])
        Bx = torch.from_numpy(magnetic_field[:, 0])
        By = torch.from_numpy(magnetic_field[:, 1])
        # shape is now (nt, nx, ny, nfields)
        data = torch.stack([u, v, Bx, By], dim=-1)
        data0 = data[0].reshape(1, nx, ny, -1).repeat(nt, 1, 1, 1)
        
        grid_t = torch.from_numpy(t).reshape(nt, 1, 1, 1).repeat(1, nx, ny, 1)
        grid_x = torch.from_numpy(x).reshape(1, nx, 1, 1).repeat(nt, 1, ny, 1)
        grid_y = torch.from_numpy(y).reshape(1, 1, ny, 1).repeat(nt, nx, 1, 1)
        
        inputs = torch.cat([grid_t, grid_x, grid_y, data0], dim=-1)
        outputs = data
        
        
        # display(inputs.shape)
        # display(outputs.shape)
        
        
        return inputs, outputs
    
    def create_dataloader(self, batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
        dataloader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        return dataloader
    

class MHDDataloaderVecPot(MHDDataloader):
    def __init__(self, dataset: Dedalus2DDataset, sub_x=1, sub_t=1, num_train=None):
        super().__init__(dataset=dataset, sub_t=sub_t, sub_x=sub_x, num_train=num_train)
        
    def __getitem__(self, index):
        fields = self.dataset[index]
        t, x, y = self.dataset.get_coords(index)
        nx = len(x)
        ny = len(y)
        nt = len(t)
        velocity = fields['velocity']
        vector_potential = fields['vector potential']
        u = torch.from_numpy(velocity[:, 0])
        v = torch.from_numpy(velocity[:, 1])
        A = torch.from_numpy(vector_potential)
        # shape is now (nt, nx, ny, nfields)
        data = torch.stack([u, v, A], dim=-1)
        data0 = data[0].reshape(1, nx, ny, -1).repeat(nt, 1, 1, 1)
        
        grid_t = torch.from_numpy(t).reshape(nt, 1, 1, 1).repeat(1, nx, ny, 1)
        grid_x = torch.from_numpy(x).reshape(1, nx, 1, 1).repeat(nt, 1, ny, 1)
        grid_y = torch.from_numpy(y).reshape(1, 1, ny, 1).repeat(nt, nx, 1, 1)
        
        inputs = torch.cat([grid_t, grid_x, grid_y, data0], dim=-1)
        outputs = data
        
        
        # display(inputs.shape)
        # display(outputs.shape)
        
        
        return inputs, outputs
    
    

if __name__ == "__main__":
    dataset = Dedalus2DDataset(data_path='mhd_data/simulation_outputs', output_names='output-????',field_names=['magnetic field', 'velocity', 'vector potential'])
    dataloader = MHDDataloader(dataset)
    dataloader = MHDDataloaderVecPot(dataset)
    
    display(dataset[0])
    # dataloader[0]