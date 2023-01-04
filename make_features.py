import multiprocessing
import os

import Levenshtein
import numpy as np
from plotly.offline import init_notebook_mode
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import torch

#init_notebook_mode(connected=True)
import glob
from scipy.stats import spearmanr
from pprint import pprint

import plotly.express as px
import torch.nn as nn
import pandas as pd
from scipy.stats import rankdata
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from collections import defaultdict
import copy
from torch.optim import AdamW

from Logger import CSVLogger

from Network import *
from Dataset import *

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0',  help='which gpu to use')
    parser.add_argument('--epochs', type=int, default=120, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=2048, help='size of each batch during training')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight dacay used in optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--nfolds', type=int, default=10, help='number of cross validation folds')
    parser.add_argument('--fold', type=int, default=0, help='which fold to train')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--nlayers', type=int, default=1, help='nlayers')
    opts = parser.parse_args()
    return opts

args=get_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


MULTIPROCESSING = False
BOXSIZE = 16
VOXELSIZE = 1
N_FOLDS = 10
MODELS_PATH = 'models'
DEBUG = True
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DESTABILIZING_MUTATIONS_ONLY = True
AUGMENT_DESTABILIZING_MUTATIONS = False
EARLY_STOPPING_PATIENCE = 30
IS_DDG_TARGET = True
WITH_PUCCI_SOURCE = True
WITH_KAGGLE_DDG_SOURCE = True

# switchers
TRAIN = True
WANDB_TRAIN_PROJECT = 'ThermoNetV2-train'
WANDB_TRAIN_NAME = 'thermonetv2-7633-v2'

OPTUNA = False
OPTUNA_WANDB_PROJECT = "ThermoNetV2-Optuna"
OPTUNA_TRIALS = 400

WANDB_SWEEP = False
WANDB_SWEEP_PROJECT = 'ThermoNetV2-sweep'

SUBMISSION = True

NUM_WORKERS=16


DEFAULT_PARAMS = {
    'SiLU': False,
    'diff_features': True,
    'LayerNorm': False,
    'GroupKFold': False,  # only use for hyperopt
    'epochs': 35,
    'AdamW': False,
}





BEST_PARAMS = {**DEFAULT_PARAMS, **{'AdamW': False,
 'C_dt_loss': 0.01,
 'OneCycleLR': False,
 'batch_size': 64,
 'AdamW_decay': 0,
 'dropout_rate': 0.06297340526648805,
 'learning_rate': 1e-2,
 'conv_layer_num': 5,
 'dropout_rate_dt': 0.3153179929570238,
 'dense_layer_size': 74.1731281147114}}



#try:
#from kaggle_secrets import UserSecretsClient
#user_secrets = UserSecretsClient()
#WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")

print('Running in Kaggle')
WILDTYPE_PDB = '../input/novozymes-enzyme-stability-prediction/wildtype_structure_prediction_af2.pdb'
PDB_PATH = '../input/thermonet-wildtype-relaxed'
TRAIN_FEATURES_PATH = '../input/thermonet-features/Q3214.npy'
TRAIN_TARGETS_PATH = ''
TEST_CSV = '../input/novozymes-enzyme-stability-prediction/test.csv'
TEST_FEATURES_PATH = '../input/thermonet-features/nesp_features.npy'
PUBLIC_SUBMISSIONS=[
    '../input/rmsd-from-molecular-dynamics/submission_rmsd.csv',     # LB: 0.507
    '../input/plldt-ddg-demask-sasa/deepddg-ddg.csv',                # LB: 0.451
    '../input/novo-esp-eli5-performant-approaches-lb-0-451/submission.csv',  # 0.451
    '../input/nesp-alphafold-getarea-exploration/submission.csv',                   # 0.407
]


TRAIN_FEATURES_DIR = '../input/14656-unique-mutations-voxel-features-pdbs/features'




os.makedirs(MODELS_PATH, exist_ok=True)





def load_data():
    print("1. Loading csv datasets")
    df = pd.read_csv(f'{TRAIN_FEATURES_DIR}/dataset.csv')

    print(df.shape)
    df.source = df.source.apply(eval)
    print(f'Total unique mutations: {len(df)}')

    df['features'] = df.apply(lambda r: f'{TRAIN_FEATURES_DIR}/{r.PDB_chain}_{r.wildtype}{r.pdb_position}{r.mutant}.npy', axis=1)
    df = df[df.features.apply(lambda v: os.path.exists(v))]

    print(df.shape)

    print(f'Total mutations with features: {len(df)}')

    if not WITH_PUCCI_SOURCE:
        df = df[df.source.apply(lambda v: v != ['pucci-proteins-appendixtable1.xlsx'])]

    if not WITH_KAGGLE_DDG_SOURCE:
        df = df[df.source.apply(lambda v: v != ['ddg-xgboost-5000-mutations-200-pdb-files-lb-0-40.csv'])]

    print(f'Total mutations after filtering: {len(df)}')

    #df.features = [np.load(f) for f in tqdm(df.features, desc="2. Loading features")]
    df.features = [None for f in tqdm(df.features, desc="2. Loading features")]

    df_train = df

    if DESTABILIZING_MUTATIONS_ONLY:
        print('Keeping destabilizing mutations only')
        df_train = df_train[((df_train.ddG < 0))  & ((df_train.dT < 0) | df_train.dT.isna())].reset_index(drop=True).copy() # best for ddG
    elif AUGMENT_DESTABILIZING_MUTATIONS:
        print('Augmenting destabilizinb mutations')
        df_pos = df_train[df_train.ddG > 0].copy()
        df_neg = df_train[df_train.ddG < 0]
        print(df_pos.shape, df_neg.shape)
        df_pos.features = df_pos.features.apply(lambda f: np.concatenate([f[7:], f[:7]], axis=0))
        df_pos.ddG = -df_pos.ddG
        df_pos.dT = -df_pos.dT
        df_train = pd.concat([df_pos, df_neg], axis=0).sample(frac=1.).reset_index(drop=True)
    return df_train

df_train = load_data()
#df_train = pd.read_csv(f'{TRAIN_FEATURES_DIR}/dataset.csv')
df_train





df_train.head()

ds_train = e3nnDataset(df_train)

features=[]
for i in tqdm(range(len(ds_train))):
    features.append(ds_train[i])

import pickle
with open('features.p','wb+') as f:
    pickle.dump(features,f)    
