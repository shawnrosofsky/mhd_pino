import os
import math
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm
import traceback

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import imageio
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from IPython.display import HTML, display


def plot_predictions_mhd(pred, true, inputs, index=0, index_t=-1, name='u', save_path=None, save_suffix=None, font_size=None, shading='auto', cmap='jet'):
    
    if font_size is not None:
        plt.rcParams.update({'font.size': font_size})
        
    
    
    Nt, Nx, Ny, Nfields = pred.shape
    u_pred = pred[index_t, ..., index]
    u_true = true[index_t, ..., index]
    u_err = u_pred - u_true
    
    initial_data = inputs[0, ..., 3:]
    u0 = initial_data[..., index]
        
    x = inputs[0, :, 0, 1]
    y = inputs[0, 0, :, 2]
    X, Y = torch.meshgrid(x, y, indexing='ij')
    t = inputs[index_t, 0, 0, 0]

    # Plot
    fig = plt.figure(figsize=(24,5))
    # fig = plt.figure()
    plt.subplot(1,4,1)

    plt.pcolormesh(X, Y, u0, cmap=cmap, shading=shading)
    plt.colorbar()
    plt.title(f'Intial Condition ${name}_0(x,y)$')
    plt.tight_layout()
    plt.axis('square')

    plt.subplot(1,4,2)
    plt.pcolormesh(X, Y, u_true, cmap=cmap, shading=shading)
    plt.colorbar()
    plt.title(f'Exact ${name}(x,y,t={t:.2f})$')
    plt.tight_layout()
    plt.axis('square')

    plt.subplot(1,4,3)
    plt.pcolormesh(X, Y, u_pred, cmap=cmap, shading=shading)
    plt.colorbar()
    plt.title(f'Predict ${name}(x,y,t={t:.2f})$')

    plt.axis('square')

    plt.tight_layout()

    plt.subplot(1,4,4)
    plt.pcolormesh(X, Y, u_pred - u_true, cmap=cmap, shading=shading)
    plt.colorbar()
    plt.title(f'Absolute Error ${name}(x,y,t={t:.2f})$')
    # plt.tight_layout()
    plt.axis('square')

    if save_path is not None:
        if save_suffix is not None:
            figure_path = f'{save_path}_{name}_{save_suffix}.png'
        else:
            figure_path = f'{save_path}_{name}.png'
        plt.savefig(figure_path, bbox_inches='tight')
    # plt.show()
    return fig

    
def plot_predictions_mhd_plotly(pred, true, inputs, index=0, index_t=-1, name='u', save_path=None, font_size=None, shading='auto', cmap='jet'):
    Nt, Nx, Ny, Nfields = pred.shape
    u_pred = pred[index_t, ..., index]
    u_true = true[index_t, ..., index]
    
    
    ic = inputs[0, ..., 3:]
    u_ic = ic[..., index]
    u_err = u_pred - u_true
    
    x = inputs[0, :, 0, 1]
    y = inputs[0, 0, :, 2]
    X, Y = torch.meshgrid(x, y, indexing='ij')
    t = inputs[index_t, 0, 0, 0]
    
    zmin = u_true.min().item()
    zmax = u_true.max().item()
    labels = {'color': name}
    
    # Initial Conditions
    title_ic = f'{name}0'
    fig_ic = px.imshow(u_ic, binary_string=False, color_continuous_scale=cmap, labels=labels, title=title_ic)
    # fig_ic = px.imshow(u_ic, x=X, y=Y, color_continuous_scale=cmap, labels=labels, title=title_ic)
    fig_ic.update_xaxes(showticklabels=False)
    fig_ic.update_yaxes(showticklabels=False)

    # Predictions
    title_pred = f'Predict {name}: t={t:.2f}'
    fig_pred = px.imshow(u_pred, binary_string=False, color_continuous_scale=cmap, labels=labels, title=title_pred)
    # fig_pred = px.imshow(u_pred, zmin=zmin, zmax=zmax, binary_string=False, color_continuous_scale=cmap, labels=labels, title=title_pred)
    # fig_pred = px.imshow(u_pred, x=X, y=Y, zmin=zmin, zmax=zmax, color_continuous_scale=cmap, labels=labels, title=title_pred)
    fig_pred.update_xaxes(showticklabels=False)
    fig_pred.update_yaxes(showticklabels=False)

    # Ground Truth
    title_true = f'Exact {name}: t={t:.2f}'
    fig_true = px.imshow(u_true, binary_string=False, color_continuous_scale=cmap, labels=labels, title=title_true)
    # fig_true = px.imshow(u_true, zmin=zmin, zmax=zmax, binary_string=False, color_continuous_scale=cmap, labels=labels, title=title_true)
    # fig_true = px.imshow(u_true, x=X, y=Y, zmin=zmin, zmax=zmax, color_continuous_scale=cmap, labels=labels, title=title_true)
    fig_true.update_xaxes(showticklabels=False)
    fig_true.update_yaxes(showticklabels=False)
    
    # Ground Truth
    title_err = f'Error {name}: t={t:.2f}'
    fig_err = px.imshow(u_err, binary_string=False, color_continuous_scale=cmap, labels=labels, title=title_err)
    # fig_err = px.imshow(u_err, x=X, y=Y, color_continuous_scale=cmap, labels=labels, title=title_err)
    fig_err.update_xaxes(showticklabels=False)
    fig_err.update_yaxes(showticklabels=False)
    
    return fig_ic, fig_pred, fig_true, fig_err