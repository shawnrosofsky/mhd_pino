from IPython.display import display, HTML
import argparse
import yaml
import os
import math
import torch
import h5py
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
# from functorch import vmap, grad
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
from utils.plot_utils import plot_predictions_mhd, plot_predictions_mhd_plotly
from importlib import reload
import imageio
import wandb

dtype = torch.float
# dtype = torch.double
torch.set_default_dtype(dtype)
default_dtype = torch.get_default_dtype()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments for training MHD PINO with Vector Potential')
    parser.add_argument('-i', '--input_dir', type=str, default='', help='Directory where the training data is stored')
    parser.add_argument('-y', '--config_file', type=str, default='config/mhd_vec_pot-0000.yaml', help='Path to the config file')
    parser.add_argument('-c', '--ckpt_path', type=str, default='', help='Directory to save checkpoints')
    parser.add_argument('-l', '--load_ckpt', type=str, default='', help='Checkpoint to load wieghts')
    parser.add_argument('-o','--output_dir', type=str, default='outputs', help='Directory to store outputs')
    parser.add_argument('-d','--wandb_dir', type=str, default='', help='wandb_directory')
    parser.add_argument('-p','--wandb_project', type=str, default='', help='wandb_directory')
    parser.add_argument('-g','--wandb_group', type=str, default='', help='wandb_directory')
    parser.add_argument('-n','--no_wandb', action='store_true', help='turn off wandb')
    
    
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # Check if GPU is available
    # torch.backends.cudnn.benchmark = False
    # print(torch.backends.cudnn.version())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.set_default_tensor_type(torch.DoubleTensor)

    # Parse command line arguments
    args = parse_arguments()
    data_dir = args.input_dir
    config_file = args.config_file
    load_ckpt = args.load_ckpt
    ckpt_path = args.ckpt_path
    output_dir = args.output_dir
    wandb_dir = args.wandb_dir
    wandb_project = args.wandb_project
    wandb_group = args.wandb_group
    use_wandb = not args.no_wandb
    
    # Load config file
    config = load_config(config_file)
    model_params = config['model_params']
    dataset_params = config['dataset_params']
    train_loader_params = config['train_loader_params']
    val_loader_params = config['val_loader_params']
    test_loader_params = config['test_loader_params']
    loss_params = config['loss_params']
    optimizer_params = config['optimizer_params']
    train_params = config['train_params']
    wandb_params = config['wandb_params']
    model_params['non_linearity'] = get_nonlinearity(model_params['activation'])
    
    # Set parameters to config values if not defined on command line
    if not data_dir:
        data_dir = dataset_params['data_dir']
        
    if not ckpt_path:
        ckpt_path = train_params['ckpt_path']
    
    if not wandb_dir:
        wandb_dir = wandb_params['wandb_dir']
    
    if not wandb_project:
        wandb_project = wandb_params['wandb_project']
    
    if not wandb_group:
        wandb_group = wandb_params['wandb_group']
    
    
    
    
    
    # Construct dataloaders
    dataset_train = Dedalus2DDataset(data_dir, output_names=dataset_params['output_names'], field_names=dataset_params['field_names'], num_train=dataset_params['num_train'], num_test=dataset_params['num_test'], use_train=True)
    dataset_val = Dedalus2DDataset(data_dir, output_names=dataset_params['output_names'], field_names=dataset_params['field_names'], num_train=dataset_params['num_train'], num_test=dataset_params['num_test'], use_train=False)
    
    mhd_dataloader_train = MHDDataloaderVecPot(dataset_train, sub_x=dataset_params['sub_x'], sub_t=dataset_params['sub_t'], ind_x=dataset_params['ind_x'], ind_t=dataset_params['ind_t'])
    mhd_dataloader_val = MHDDataloaderVecPot(dataset_val, sub_x=dataset_params['sub_x'], sub_t=dataset_params['sub_t'], ind_x=dataset_params['ind_x'], ind_t=dataset_params['ind_t'])

    dataloader_train = mhd_dataloader_train.create_dataloader(batch_size=train_loader_params['batch_size'], shuffle=train_loader_params['shuffle'], num_workers=train_loader_params['num_workers'], pin_memory=train_loader_params['pin_memory'])
    dataloader_val = mhd_dataloader_val.create_dataloader(batch_size=val_loader_params['batch_size'], shuffle=val_loader_params['shuffle'], num_workers=val_loader_params['num_workers'], pin_memory=val_loader_params['pin_memory'])
    # dataloader_train = mhd_dataloader_train.create_dataloader(**train_loader_params)
    # dataloader_val = mhd_dataloader_val.create_dataloader(**val_loader_params)
    
    
    # Construct model
    # model = FNN3d(modes1=model_params['modes1'],
    #               modes2=model_params['modes2'],
    #               modes3=model_params['modes3'],
    #               fc_dim=model_params['fc_dim'],
    #               layers=model_params['layers'], 
    #               in_dim=model_params['in_dim'],
    #               out_dim=model_params['out_dim'],
    #               activation=model_params['activation'],
    #               pad_x=model_params['pad_x'],
    #               pad_y=model_params['pad_y'],
    #               pad_z=model_params['pad_z']).to(device)
    # model = FNN3d(**model_params).to(device)
    model = FactorizedFNO3d(**model_params,
                            joint_factorization=True, 
                            # rank=0.5, #1.0, 
                            # factorization='cp', 
                            fixed_rank_modes=False,
                            Block=None,
                            verbose=False, 
                            fft_contraction='complex',
                            fft_norm='backward',
                            mlp=False,
                            decomposition_kwargs=dict()).to(device)
    
    
    # Construct optimizer and scheduler
    optimizer = Adam(model.parameters(), betas=optimizer_params['betas'], lr=optimizer_params['lr'])
    # optimizer = AdamW(model.parameters(), betas=optimizer_params['betas'], lr=optimizer_params['lr'], weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=optimizer_params['milestones'], gamma=optimizer_params['gamma'])
    
    # Construct Loss class
    mhd_loss = LossMHDVecPot(**loss_params)
    
    # Load model from checkpoint (if exists)
    if load_ckpt:
        load_checkpoint(model, ckpt_path=load_ckpt, optimizer=None, device=device)
    
    if use_wandb:
        # Initialize wandb
        wandb.init(dir=wandb_dir,
                   project=wandb_project,
                   group=wandb_group, 
                   config=config)
    
# Training Loop
    epochs = train_params['epochs']
    ckpt_freq = train_params['ckpt_freq']
    # pbar_epoch = tqdm.tqdm(range(epochs), dynamic_ncols=True, smoothing=0.1)
    names = dataset_params['fields']
    for e in range(epochs):
        print(f'Epoch: {e}')
        
        
        # Train Loop
        model.train()
        train_loss = []
        train_loss_dict = {}
        print('Training:')
        pbar_train = tqdm.tqdm(dataloader_train, dynamic_ncols=True, smoothing=0.1)
        for i, (inputs, outputs) in enumerate(pbar_train):
            inputs = inputs.type(dtype).to(device)
            outputs = outputs.type(dtype).to(device)
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
                inputs = inputs.type(torch.FloatTensor).to(device)
                outputs = outputs.type(torch.FloatTensor).to(device)
                
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