#!/bin/bash
RE=500; N=128; WEIGHT=0; GPU=0; 
CUDA_VISIBLE_DEVICES=$GPU nohup python train_mhd_vec_pot_tfno.py -y config/mhd_vec_pot_Re${RE}_N${N}_tfno_pde_weight_${WEIGHT}.yaml -l checkpoints/MHDVecPot_tfno/MHDVecPot_PINO_tfno.pt &> logs/nohup_RE${RE}_N${N}_pde_weight_${WEIGHT}.out &
RE=500; N=128; WEIGHT=2; GPU=1; 
CUDA_VISIBLE_DEVICES=$GPU nohup python train_mhd_vec_pot_tfno.py -y config/mhd_vec_pot_Re${RE}_N${N}_tfno_pde_weight_${WEIGHT}.yaml -l checkpoints/MHDVecPot_tfno/MHDVecPot_PINO_tfno.pt &> logs/nohup_RE${RE}_N${N}_pde_weight_${WEIGHT}.out &
RE=500; N=128; WEIGHT=5; GPU=2; 
CUDA_VISIBLE_DEVICES=$GPU nohup python train_mhd_vec_pot_tfno.py -y config/mhd_vec_pot_Re${RE}_N${N}_tfno_pde_weight_${WEIGHT}.yaml -l checkpoints/MHDVecPot_tfno/MHDVecPot_PINO_tfno.pt &> logs/nohup_RE${RE}_N${N}_pde_weight_${WEIGHT}.out &
RE=500; N=128; WEIGHT=10; GPU=3; 
CUDA_VISIBLE_DEVICES=$GPU nohup python train_mhd_vec_pot_tfno.py -y config/mhd_vec_pot_Re${RE}_N${N}_tfno_pde_weight_${WEIGHT}.yaml -l checkpoints/MHDVecPot_tfno/MHDVecPot_PINO_tfno.pt &> logs/nohup_RE${RE}_N${N}_pde_weight_${WEIGHT}.out &
