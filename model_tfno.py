from IPython.display import display
import argparse
import yaml
import os
import math
import torch
import h5py
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from functorch import vmap, grad
from models import FNN3d, FactorizedFNO3d
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import plotly
import numpy as np
import traceback
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import tqdm
from losses import LpLoss, LossMHD, LossMHDVecPot
from dataloaders import Dedalus2DDataset, MHDDataloader, MHDDataloaderVecPot

from utils.adam import Adam
from torch.optim import AdamW
from utils.my_random_fields import GRF_Mattern
from utils.utils import load_checkpoint, load_config, save_checkpoint, update_config, get_nonlinearity
from utils.plot_utils import plot_predictions_mhd, plot_predictions_mhd_plotly, plot_spectra_mhd
from importlib import reload
import imageio
import wandb

import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# dtype = torch.float
# # dtype = torch.double
# torch.set_default_dtype(dtype)
# default_dtype = torch.get_default_dtype()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model_tfno(object):
    def __init__(self, config_file, ckpt_path='', data_dir='', load_ckpt=True, use_wandb=False, wandb_dir='', wandb_project='', wandb_group='', dtype=torch.float, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        # Define parameters
        self.config_file = config_file
        self.config = config = load_config(config_file)
        self.config = config = load_config(config_file)
        self.model_params = model_params = config['model_params']
        self.dataset_params = dataset_params = config['dataset_params']
        self.train_loader_params = train_loader_params = config['train_loader_params']
        self.val_loader_params = val_loader_params = config['val_loader_params']
        self.test_loader_params = test_loader_params = config['test_loader_params']
        self.loss_params = loss_params = config['loss_params']
        self.optimizer_params = optimizer_params = config['optimizer_params']
        self.train_params = train_params = config['train_params']
        self.test_params = test_params = config['test_params']
        self.wandb_params = wandb_params = config['wandb_params']
        self.use_wandb = use_wandb
        if not data_dir:
            self.data_dir = data_dir = dataset_params['data_dir']
        if not ckpt_path:
            self.ckpt_path = ckpt_path = test_params['ckpt']
        if not wandb_dir:
            self.wandb_dir = wandb_dir = wandb_params['wandb_dir']
        if not wandb_project:
            self.wandb_project = wandb_project = wandb_params['wandb_project']
        if not wandb_group:
            self.wandb_group = wandb_group = wandb_params['wandb_group']
        self.names = names = dataset_params['fields']
        self.dtype = dtype

        
        # Define dataloaders
        self.dataset_train = dataset_train = Dedalus2DDataset(data_dir, output_names=dataset_params['output_names'], field_names=dataset_params['field_names'], num_train=dataset_params['num_train'], num_test=dataset_params['num_test'], use_train=True)
        self.dataset_val = dataset_val = Dedalus2DDataset(data_dir, output_names=dataset_params['output_names'], field_names=dataset_params['field_names'], num_train=dataset_params['num_train'], num_test=dataset_params['num_test'], use_train=False)

        self.mhd_dataloader_train = mhd_dataloader_train = MHDDataloaderVecPot(dataset_train, sub_x=dataset_params['sub_x'], sub_t=dataset_params['sub_t'], ind_x=dataset_params['ind_x'], ind_t=dataset_params['ind_t'])
        self.mhd_dataloader_val = mhd_dataloader_val = MHDDataloaderVecPot(dataset_val, sub_x=dataset_params['sub_x'], sub_t=dataset_params['sub_t'], ind_x=dataset_params['ind_x'], ind_t=dataset_params['ind_t'])

        self.dataloader_train = dataloader_train = mhd_dataloader_train.create_dataloader(batch_size=train_loader_params['batch_size'], shuffle=train_loader_params['shuffle'], num_workers=train_loader_params['num_workers'], pin_memory=train_loader_params['pin_memory'])
        self.dataloader_val = dataloader_val = mhd_dataloader_val.create_dataloader(batch_size=val_loader_params['batch_size'], shuffle=val_loader_params['shuffle'], num_workers=val_loader_params['num_workers'], pin_memory=val_loader_params['pin_memory'])
        
        # Construct Model
        self.model = model = FactorizedFNO3d(**model_params, joint_factorization=True,  fixed_rank_modes=False, Block=None, verbose=False,  fft_contraction='complex', fft_norm='backward', mlp=False, decomposition_kwargs=dict()).to(device)
        
        # Construct Optimizer
        # self.optimizer = optimizer = Adam(model.parameters(), betas=optimizer_params['betas'], lr=optimizer_params['lr'])
        self.optimizer = optimizer = AdamW(model.parameters(), betas=optimizer_params['betas'], lr=optimizer_params['lr'], weight_decay=0.1)
        self.scheduler = scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=optimizer_params['milestones'], gamma=optimizer_params['gamma'])
        
        # Define Loss
        self.mhd_loss = mhd_loss = LossMHDVecPot(**loss_params)
        
        # Load checkpoint if it exists
        if load_ckpt:
            load_checkpoint(model, ckpt_path=ckpt_path, optimizer=None, device=device)

        
    # Train Function
    def train(self, epochs=None, ckpt_freq=None):
        device = self.device
        config_file = self.config_file
        config = self.config
        config = self.config
        model_params = self.model_params
        dataset_params = self.dataset_params
        train_loader_params = self.train_loader_params
        val_loader_params = self.val_loader_params
        test_loader_params = self.test_loader_params
        loss_params = self.loss_params
        optimizer_params = self.optimizer_params
        train_params = self.train_params
        test_params = self.test_params
        wandb_params = self.wandb_params
        use_wandb = self.use_wandb
        data_dir = self.data_dir
        ckpt_path = self.ckpt_path
        wandb_dir = self.wandb_dir
        wandb_project = self.wandb_project
        wandb_group = self.wandb_group
        names = self.names
        dataset_train = self.dataset_train
        dataset_val = self.dataset_val
        mhd_dataloader_train = self.mhd_dataloader_train
        mhd_dataloader_val = self.mhd_dataloader_val
        dataloader_train = self.dataloader_train
        dataloader_val = self.dataloader_val
        model = self.model
        optimizer = self.optimizer
        optimizer = self.optimizer
        scheduler = self.scheduler
        mhd_loss = self.mhd_loss
        dtype = self.dtype
        
        if epochs is None:
            epochs = train_params['epochs']
        ckpt_freq = train_params['ckpt_freq']
        ckpt_freq = train_params['ckpt_freq']
        
        for e in range(epochs):
            print(f'Epoch: {e}')
            
            
            # Train Loop
            model.train()
            train_loss = []
            train_loss_dict = {}
            print('Training:')
            pbar_train = tqdm.tqdm(dataloader_train, dynamic_ncols=True, smoothing=0.1)
            for i, (inputs, outputs) in enumerate(pbar_train):
                inputs = inputs.type(torch.FloatTensor).to(device)
                outputs = outputs.type(torch.FloatTensor).to(device)
                # Zero Gradients
                optimizer.zero_grad()
                # Compute Predictions
                pred = model(inputs)
                # Compute Loss
                loss, loss_dict = mhd_loss(pred, outputs, inputs, return_loss_dict=True)
                # Compute Gradients for Back Propagation
                loss.backward()
                # Update Weights
                optimizer.step()
                
                # Add losses to running sum
                train_loss.append(loss.item())
                for key in loss_dict:
                    if key in train_loss_dict:
                        train_loss_dict[key] += loss_dict[key].item()
                    else:
                        train_loss_dict[key] = loss_dict[key].item()

                # Update progress bar
                pbar_train.set_description(f'epoch {e}; batch {i}; train_loss {np.mean(train_loss):.5f}')
            # Get the loss dict in proper format for wandb
            train_loss_dict = {key: train_loss_dict[key]/len(dataloader_train) for key in train_loss_dict}
            scheduler.step()
            
            # Val loop
            model.eval()
            val_loss = []
            val_loss_dict = {}
            plot_count = 0
            plot_dict = {name: {} for name in names}
            print('Validation:')
            pbar_val = tqdm.tqdm(dataloader_val, dynamic_ncols=True, smoothing=0.1)
            with torch.no_grad():
                for i, (inputs, outputs) in enumerate(pbar_val):
                    inputs = inputs.type(dtype).to(device)
                    outputs = outputs.type(dtype).to(device)
                    
                    # Compute Predictions
                    pred = model(inputs).reshape(outputs.shape)
                    # Compute Loss
                    loss, loss_dict = mhd_loss(pred, outputs, inputs, return_loss_dict=True)
                    
                    # Add losses to running sum
                    val_loss.append(loss.item())
                    for key in loss_dict:
                        if key in val_loss_dict:
                            val_loss_dict[key] += loss_dict[key].item()
                        else:
                            val_loss_dict[key] = loss_dict[key].item()

                    # Update progress bar
                    pbar_val.set_description(f'epoch {e}; batch {i}; val_loss {np.mean(val_loss):.5f}')
                
                    # Get prediction plots to log for wandb
                    # Do for number of batches specified in the config file
                    if (i < wandb_params['wandb_num_plots']) and (e % wandb_params['wandb_plot_freq'] == 0):
                        # Add all predictions in batch
                        for j, _ in enumerate(pred):
                            # Make plots for each field
                            for index, name in enumerate(names):
                                # Generate figure
                                # fig = plot_predictions_mhd(pred[j].cpu(), outputs[j].cpu(), inputs[j].cpu(), index=index, name=name)
                                figs = plot_predictions_mhd_plotly(pred[j].cpu(), outputs[j].cpu(), inputs[j].cpu(), index=index, name=name)
                                # Add figure to plot dict
                                plot_dict[name] = {f'{plot_type}-{plot_count}': wandb.Html(plotly.io.to_html(fig)) for plot_type, fig in zip(wandb_params['wandb_plot_types'], figs)}
                                # for plot_type in wandb_params['wandb_plot_types']:
                                #     plot_dict[name][f'{plot_type}-{}'][f'{name}-{plot_count}'] = fig
                            plot_count += 1
            # Get the loss dict in proper format for wandb
            val_loss_dict = {key: val_loss_dict[key]/len(dataloader_val) for key in val_loss_dict}
            
            if use_wandb:
                wandb_dict = {'epoch': e,
                            'train': train_loss_dict,
                            'val': val_loss_dict}
                if e % wandb_params['wandb_plot_freq'] == 0:
                    wandb_dict['plots'] = plot_dict
                
                wandb.log(wandb_dict)
                # wandb.log({'epoch': e,
                #            'train': train_loss_dict,
                #            'val': val_loss_dict,
                #            'plots': plot_dict})
            
            if e % ckpt_freq == 0:
                ckpt_path_epoch = ckpt_path.replace('.pt', f'_{e:03d}.pt')
                save_checkpoint(ckpt_path_epoch, model, optimizer)
        
        print('Finished Training')
        
        save_checkpoint(ckpt_path, model, optimizer)

        if use_wandb:
            wandb.finish()
            
    # Def eval function
    def evaluate(self, dataloader_val=None):
        model = self.model
        mhd_loss = self.mhd_loss
        device = self.device
        dtype = self.dtype
        
        if dataloader_val is None:
            dataloader_val = self.dataloader_val

        # Val loop
        model.eval()
        val_loss = []
        val_loss_dict = {}
        print('Validation:')
        pbar_val = tqdm.tqdm(dataloader_val, dynamic_ncols=True, smoothing=0.1)
        with torch.no_grad():
            for i, (inputs, outputs) in enumerate(pbar_val):
                inputs = inputs.type(dtype).to(device)
                outputs = outputs.type(dtype).to(device)
                
                # Compute Predictions
                pred = model(inputs).reshape(outputs.shape)
                # Compute Loss
                loss, loss_dict = mhd_loss(pred, outputs, inputs, return_loss_dict=True)
                
                # Add losses to running sum
                val_loss.append(loss.item())
                for key in loss_dict:
                    if key in val_loss_dict:
                        val_loss_dict[key] += loss_dict[key].item()
                    else:
                        val_loss_dict[key] = loss_dict[key].item()

                # Update progress bar
                pbar_val.set_description(f'val_loss {np.mean(val_loss):.5f}')
            
        # Get the loss dict in proper format for wandb
        val_loss_dict = {key: val_loss_dict[key]/len(dataloader_val) for key in val_loss_dict}
        
        return val_loss_dict
    
    def gen_preds(self, dataloader_val=None, return_test_data=False):
        model = self.model
        mhd_loss = self.mhd_loss
        names = self.names
        device = self.device
        dtype = self.dtype
        
        if dataloader_val is None:
            dataloader_val = self.dataloader_val

        preds = []
        test_x = []
        test_y = []
        # Val loop
        model.eval()
        with torch.no_grad():
            for i, (inputs, outputs) in enumerate(dataloader_val):
                inputs = inputs.type(dtype).to(device)
                outputs = outputs.type(dtype).to(device)
                pred = model(inputs).reshape(outputs.shape)
                for j, _ in enumerate(pred):
                    preds.append(pred[j].cpu())
                    if return_test_data:
                        test_x.append(inputs[j].cpu())
                        test_y.append(outputs[j].cpu())
                    
        
        if return_test_data:
            return preds, test_x, test_y
        else:
            return preds
    
    def plot_predictions(self, pred, true, inputs, key=0, index_t=-1, save_path=None, font_size=None, shading='gouraud', cmap='jet'):
        names = self.names
        for index, name in enumerate(names):
            plot_predictions_mhd(pred[key], true[key], inputs[key], index=index, index_t=index_t, name=name, save_path=save_path, save_suffix=key, font_size=font_size, shading=shading, cmap=cmap)
            
    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
    
    def load_checkpoint(self, ckpt_path=''):
        model = self.model
        device = self.device
        if ckpt_path:
            self.ckpt_path = ckpt_path
        load_checkpoint(model, ckpt_path=ckpt_path, optimizer=None, device=device)
    
    def save_ckeckpoint(self, ckpt_path=''):
        model = self.model
        optimizer = self.optimizer
        device = self.device
        if not ckpt_path:
            ckpt_path = self.ckpt_path
        save_checkpoint(ckpt_path, model, optimizer)



    
    def calc_spectra(self, pred, true, key=0, nbins=None):
        device = self.device
        
        u_pred = pred[key][..., 0].to(device)
        v_pred = pred[key][..., 1].to(device)
        A_pred = pred[key][..., 2].to(device)
        
        u_true = true[key][..., 0].to(device)
        v_true = true[key][..., 1].to(device)
        A_true = true[key][..., 2].to(device)
        
        
        
        
        pred_spectra_kin, k = self.calc_kin_spectra(u_pred, v_pred, nbins=nbins)
        true_spectra_kin, _ = self.calc_kin_spectra(u_true, v_true, nbins=nbins)
       
        pred_spectra_mag, _ = self.calc_mag_spectra(A_pred, nbins=nbins)
        true_spectra_mag, _ = self.calc_mag_spectra(A_true, nbins=nbins)
        
        pred_spectra_kin = pred_spectra_kin.cpu()
        true_spectra_kin = true_spectra_kin.cpu()
        pred_spectra_mag = pred_spectra_mag.cpu()
        true_spectra_mag = true_spectra_mag.cpu()
        k = k.cpu()
        
        return pred_spectra_kin, true_spectra_kin, pred_spectra_mag, true_spectra_mag, k
        
        
    def calc_kin_spectra(self, u, v, nbins=None):
        device = self.device
        loss_params = self.loss_params
        
        nt = u.size(0)
        nx = u.size(1)
        ny = u.size(2)
        
        Lx = loss_params['Lx']
        Ly = loss_params['Ly']
        
        k_max = nx//2
        
        k_x = 2*np.pi/Lx * torch.cat([torch.arange(start=0, end=k_max, step=1, device=device),
                                      torch.arange(start=-k_max, end=0, step=1, device=device)], 0).reshape(nx, 1).repeat(1, ny).reshape(1,nx,ny)
        k_y = 2*np.pi/Ly * torch.cat([torch.arange(start=0, end=k_max, step=1, device=device),
                                      torch.arange(start=-k_max, end=0, step=1, device=device)], 0).reshape(1, ny).repeat(nx, 1).reshape(1,nx,ny)
        
        k_x = torch.fft.fftshift(k_x, dim=[1, 2])
        k_y = torch.fft.fftshift(k_y, dim=[1, 2])
        
        u_h = torch.fft.fftshift(torch.fft.fftn(u, dim=[1, 2]) / (nx*ny), dim=[1, 2])
        v_h = torch.fft.fftshift(torch.fft.fftn(v, dim=[1, 2]) / (nx*ny), dim=[1, 2])
        
        uk = u_h*torch.conj(u_h)
        vk = v_h*torch.conj(v_h)
        
        E2D = uk + vk
        E2D = torch.abs(E2D)
        K2D = torch.sqrt(k_x**2 + k_y**2)        
        
        if nbins is None:
            nbins = nx
        
        dkx = torch.pi/Lx * nx/nbins
        dky = torch.pi/Ly * ny/nbins
        dk = math.sqrt(dkx**2 + dky**2)
        
        k = torch.arange(1,nbins+1, device=device)*dk
        E = torch.zeros(nt, nbins, device=device)
        for i in range(nbins):
            k_active = abs(k[i]-K2D)<0.5*dk
            E_bin = torch.sum(k_active*E2D, dim=[1, 2])
            E[:, i] += E_bin
        return E, k
        
    
    def calc_mag_spectra(self, A, nbins=None):
        device = self.device
        loss_params = self.loss_params
        
        nt = A.size(0)
        nx = A.size(1)
        ny = A.size(2)
        
        Lx = loss_params['Lx']
        Ly = loss_params['Ly']
        
        k_max = nx//2
        
        k_x = 2*np.pi/Lx * torch.cat([torch.arange(start=0, end=k_max, step=1, device=device),
                                      torch.arange(start=-k_max, end=0, step=1, device=device)], 0).reshape(nx, 1).repeat(1, ny).reshape(1,nx,ny)
        k_y = 2*np.pi/Ly * torch.cat([torch.arange(start=0, end=k_max, step=1, device=device),
                                      torch.arange(start=-k_max, end=0, step=1, device=device)], 0).reshape(1, ny).repeat(nx, 1).reshape(1,nx,ny)
        
        k_x = torch.fft.fftshift(k_x, dim=[1, 2])
        k_y = torch.fft.fftshift(k_y, dim=[1, 2])
        
        A_h = torch.fft.fftshift(torch.fft.fftn(A, dim=[1, 2]) / (nx*ny), dim=[1, 2])
        
        Ax_h = self.Du_i(A_h, k_x)
        Ay_h = self.Du_i(A_h, k_y)
        
        Bx_h = Ay_h
        By_h = -Ax_h        
        
        Bx_k = Bx_h*torch.conj(Bx_h)
        By_k = By_h*torch.conj(By_h)
        
        E2D = Bx_k + By_k
        E2D = torch.abs(E2D)
        K2D = torch.sqrt(k_x**2 + k_y**2)  
        
        if nbins is None:
            nbins = nx
        
        dkx = torch.pi/Lx * nx/nbins
        dky = torch.pi/Ly * ny/nbins
        dk = math.sqrt(dkx**2 + dky**2)
        
        k = torch.arange(1,nbins+1, device=device)*dk
        E = torch.zeros(nt, nbins, device=device)
        for i in range(nbins):
            k_active = abs(k[i]-K2D)<0.5*dk
            E_bin = torch.sum(k_active*E2D, dim=[1, 2])
            E[:, i] += E_bin
        return E, k
        
    def plot_spectra(self, k, pred_spectra_kin, true_spectra_kin, pred_spectra_mag, true_spectra_mag, index_t=-1, name='Re100', save_path=None, save_suffix=None, font_size=None, style_kin_pred='b-', style_kin_true='k-', style_mag_pred='b--', style_mag_true='k--', xmin=0, xmax=200, ymin=1e-10, ymax=None):
        plot_spectra_mhd(k, pred_spectra_kin, true_spectra_kin, pred_spectra_mag, true_spectra_mag, index_t=index_t, name=name, save_path=save_path, save_suffix=save_suffix, font_size=font_size, style_kin_pred=style_kin_pred, style_kin_true=style_kin_true, style_mag_pred=style_mag_pred, style_mag_true=style_mag_true, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        
    def calc_and_plot_spectra(self, pred, true, key=0, index_t=-1, nbins=None, name='Re100', save_path=None, save_suffix=None, font_size=None, style_kin_pred='b-', style_kin_true='k-', style_mag_pred='b--', style_mag_true='k--', xmin=0, xmax=200, ymin=1e-10, ymax=None, return_spectra=True):
        
        pred_spectra_kin, true_spectra_kin, pred_spectra_mag, true_spectra_mag, k = self.calc_spectra(pred, true, key=key, nbins=nbins)
        
        self.plot_spectra(k, pred_spectra_kin, true_spectra_kin, pred_spectra_mag, true_spectra_mag, index_t=index_t, name=name, save_path=save_path, save_suffix=save_suffix, font_size=font_size, style_kin_pred=style_kin_pred, style_kin_true=style_kin_true, style_mag_pred=style_mag_pred, style_mag_true=style_mag_true, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        
        if return_spectra:
            return pred_spectra_kin, true_spectra_kin, pred_spectra_mag, true_spectra_mag, k
        
        
    
    def Du_i(self, u_h, k_i):
        u_i_h = (1j*k_i) * u_h
        return u_i_h
    
    def Du_ij(self, u_h, k_i, k_j):
        u_ij_h = (1j*k_i) * (1j*k_j) * u_h
        return u_ij_h
    
    def Du_ii(self, u_h, k_i):
        u_ii_h = self.Du_ij(u_h, k_i, k_i)
        return u_ii_h