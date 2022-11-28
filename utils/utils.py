import os
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import traceback

def load_checkpoint_new(model, ckpt_path, optimizer=None, device=None):
    try:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # ckpt = torch.load(ckpt_path, map_location=device)

        my_model_dict = model.state_dict()
        ckpt = torch.load(ckpt_path, map_location=device)

        part_load = {}
        match_size = 0
        nomatch_size = 0
        for k in ckpt.keys():
            value = ckpt[k]
            if k in my_model_dict and my_model_dict[k].shape == value.shape:
                # print("loading ", k)
                match_size += 1
                part_load[k] = value
            else:
                nomatch_size += 1

        print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

        my_model_dict.update(part_load)
        model.load_state_dict(my_model_dict)
        print(f'Weights loaded from {ckpt_path}')
        if optimizer is not None:
            try:
                optimizer.load_state_dict(ckpt['optim'])
                print('Optimizer loaded from %s' % ckpt_path)
            except: traceback.print_exc()
    except:
        traceback.print_exc()


    return model

def load_checkpoint(model, ckpt_path, optimizer=None, device=None):
    try:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
        if optimizer is not None:
            try:
                optimizer.load_state_dict(ckpt['optim'])
                print('Optimizer loaded from %s' % ckpt_path)
            except: traceback.print_exc()
            
    except:
        traceback.print_exc()

def save_checkpoint(ckpt_path, model, optimizer=None):
    ckpt_dir, ckpt_name = os.path.split(ckpt_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()

    if optimizer is not None:
        optim_dict = optimizer.state_dict()
    else:
        optim_dict = 0.0

    torch.save({
        'model': model_state_dict,
        'optim': optim_dict
    }, ckpt_path)
    print(f'Checkpoint is saved at {ckpt_path}')


def update_config(config, file):
    with open(file, 'w') as f:
        config_updated = yaml.dump(config, f)
        
def load_config(file):
    with open(file, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    return config




def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def set_grad(tensors, flag=True):
    for p in tensors:
        p.requires_grad = flag


def zero_grad(params):
    '''
    set grad field to 0
    '''
    if isinstance(params, torch.Tensor):
        if params.grad is not None:
            params.grad.zero_()
    else:
        for p in params:
            if p.grad is not None:
                p.grad.zero_()


def count_params(net):
    count = 0
    for p in net.parameters():
        count += p.numel()
    return count

def get_nonlinearity(activation):
    if activation =='tanh':
        nonlinearity = F.tanh
    elif activation == 'gelu':
        nonlinearity = F.gelu
    elif activation == 'relu':
        nonlinearity == F.relu
    elif activation == 'linear':
        nonlinearity == F.linear
    elif activation == 'bilinear':
        nonlinearity == F.bilinear
    elif activation == 'hardtanh':
        nonlinearity == F.hardtanh
    elif activation == 'relu6':
        nonlinearity == F.relu6
    elif activation == 'elu':
        nonlinearity == F.elu
    elif activation == 'selu':
        nonlinearity == F.selu
    elif activation == 'celu':
        nonlinearity == F.celu
    elif activation == 'leaky_relu':
        nonlinearity == F.leaky_relu
    elif activation == 'prelu':
        nonlinearity == F.prelu
    elif activation == 'rrelu':
        nonlinearity == F.rrelu
    elif activation == 'silu':
        nonlinearity == F.silu
    elif activation == 'mish':
        nonlinearity == F.mish
    else:
        nonlinearity = activation
    return nonlinearity
        
    