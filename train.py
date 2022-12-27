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
    parser.add_argument('--models_path', type=str, default='models',  help='path to save models')
    parser.add_argument('--destabilizing_mutations_only', action='store_true', help='use destablizing mutations only or not')
    opts = parser.parse_args()
    return opts

args=get_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


DEBUG = True
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
AUGMENT_DESTABILIZING_MUTATIONS = False
EARLY_STOPPING_PATIENCE = 30
IS_DDG_TARGET = True
WITH_PUCCI_SOURCE = True
WITH_KAGGLE_DDG_SOURCE = True

# switchers
WANDB_TRAIN_PROJECT = 'ThermoNetV2-train'
WANDB_TRAIN_NAME = 'thermonetv2-7633-v2'


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





TRAIN_FEATURES_DIR = '../input/14656-unique-mutations-voxel-features-pdbs/features'




os.makedirs(args.models_path, exist_ok=True)





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

    if args.destabilizing_mutations_only:
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


import pickle
try:
    with open('features.p','rb') as f:
        features=pickle.load(f)
except:
    features=None



df_train.head()




def evaluate(model, dl_val, params):
    criterion = nn.MSELoss()
    model.eval()
    losses = []
    ddg_preds = []
    dt_preds = []
    ddg_losses = []
    dt_losses = []
    with torch.no_grad():
        for batch in tqdm(dl_val, desc='Eval', disable=True):

            for key in batch:
                batch[key]=batch[key].to(DEVICE)

            ddg, dt = batch['ddg'], batch['ddt']
            ddg_pred, dt_pred = model(batch)
            ddg_preds.append(ddg_pred.cpu().numpy())
            dt_preds.append(dt_pred.cpu().numpy())
            ddg = ddg.to(DEVICE)
            dt = dt.to(DEVICE)
            not_nan_ddg = ~torch.isnan(ddg)
            ddg_loss = criterion(ddg[not_nan_ddg], ddg_pred[not_nan_ddg])

            not_nan_dt = ~torch.isnan(dt)
            dt_loss = criterion(dt[not_nan_dt], dt_pred[not_nan_dt])

            loss = torch.stack([ddg_loss, dt_loss * params['C_dt_loss']])
            loss = loss[~torch.isnan(loss)].sum()
            if not np.isnan(loss.item()):
                losses.append(loss.item())
            if not np.isnan(ddg_loss.item()):
                ddg_losses.append(ddg_loss.item())
            if not np.isnan(dt_loss.item()):
                dt_losses.append(dt_loss.item())

    return np.mean(losses), np.mean(ddg_losses), np.mean(dt_losses), np.concatenate(ddg_preds), np.concatenate(dt_preds)



def train_model(name, dl_train, dl_val, params, logger, wandb_enabled=False, project='thermonetv2'):
    model = e3nnNetwork().to(DEVICE).double()

    # else:
    optim = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    scheduler = None
    if params['OneCycleLR']:
        from torch.optim.lr_scheduler import OneCycleLR
        scheduler = OneCycleLR(optim, max_lr=params['learning_rate'],
                                                     steps_per_epoch=len(dl_train), epochs=params['epochs'],
                                                     pct_start=0.)
    criterion = nn.MSELoss()
    best_model = None
    min_epoch = -1
    val_losses = defaultdict(lambda: [])

    run = None
    if wandb_enabled:
        run = wandb.init(project=project, name=name, mode='online' if wandb_enabled else 'disabled')

    with tqdm(range(params['epochs']), desc='Epoch') as prog:
        min_loss = np.inf
        for epoch in prog:
            model.train()
            #print(len(dl_train))
            train_loss=[]
            #for x, ddg, dt in tqdm(dl_train, desc='Train', disable=True):
            for batch in tqdm(dl_train, desc='Train', disable=False):

                # for key in batch:
                #     print(batch[key].shape)
                # exit()

                for key in batch:
                    batch[key]=batch[key].to(DEVICE)

                ddg, dt = batch['ddg'], batch['ddt']

                ddg_pred, dt_pred=model(batch)

                #ddg_pred, dt_pred = model(x.to(DEVICE))
                ddg = ddg.to(DEVICE)
                dt = dt.to(DEVICE)
                loss = None
                any_ddg = ~torch.isnan(ddg)
                if torch.any(any_ddg):
                    loss = criterion(ddg[any_ddg], ddg_pred[any_ddg])
                #print(loss)
                any_dt = ~torch.isnan(dt)
                if torch.any(any_dt):
                    dt_loss = criterion(dt[any_dt], dt_pred[any_dt])
                    if loss is None:
                        loss = dt_loss * params['C_dt_loss']
                    else:
                        loss += dt_loss * params['C_dt_loss']
                train_loss.append(loss.item())
                #print(loss)
                optim.zero_grad()
                loss.backward()
                optim.step()
                if scheduler is not None:
                    scheduler.step()

                #break

            train_loss=np.mean(train_loss)
            eval_loss, eval_ddg_loss, eval_dt_loss = evaluate(model, dl_val, params)[:3]
            val_losses['loss'].append(eval_loss)
            val_losses['ddg_loss'].append(eval_ddg_loss)
            val_losses['dt_loss'].append(eval_dt_loss)
            if run is not None:
                run.log({'val_loss': eval_loss, 'val_ddg_loss': eval_ddg_loss, 'val_dt_loss': eval_dt_loss,
                         'lr': scheduler.get_last_lr()[0] if scheduler is not None else params['learning_rate']})
            if eval_ddg_loss < min_loss:
                min_loss = eval_ddg_loss
                min_epoch = epoch
                best_model = copy.deepcopy(model)
                fname = f'{args.models_path}/{name}.pt'
                torch.save(model.state_dict(),f'models/fold{args.fold}.pt')
            prog.set_description(
                f'Epoch: {epoch}; Train Loss: {train_loss:.02f} Val MSE:{eval_loss:.02f}; Min Val MSE:{min_loss:.02f}; ddg loss:{eval_ddg_loss:.02f}; dT loss:{eval_dt_loss:.02f}')
            logger.log([epoch,train_loss,eval_loss,eval_ddg_loss,eval_dt_loss])
            if epoch - min_epoch > EARLY_STOPPING_PATIENCE:
                print('Early stopping')
                break

    if run is not None:
        #art = wandb.Artifact("thermonet2", type="model")
        fname = f'{args.models_path}/{name}.pt'
        torch.save(model.state_dict(), fname)
        art.add_file(fname)
        run.log_artifact(art)
        run.finish()
    return best_model, val_losses


def run_train(name, params, project='thermonetv2'):
    os.makedirs(args.models_path, exist_ok=True)
    os.makedirs('logs', exist_ok=True)



    val_losses = []
    thermonet_models = []
    kfold = GroupKFold(args.nfolds)
    if params['GroupKFold']:
        groups = df_train.sequence
    else:
        groups = range(len(df_train))

    split=[]
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df_train, groups=groups)):
        split.append([fold, train_idx, val_idx])


    for fold, train_idx, val_idx in tqdm([split[args.fold]], total=1, desc="Folds"):
        exp_name = f'{name}-{fold}'
        fname = f'{args.models_path}/{exp_name}.pt'
        # ds_train = ThermoNet2Dataset(df_train.loc[train_idx])
        # ds_val = ThermoNet2Dataset(df_train.loc[val_idx])


        ds_train = e3nnDataset(df_train.loc[train_idx],[features[i] for i in train_idx])
        ds_val = e3nnDataset(df_train.loc[val_idx],[features[i] for i in val_idx])

        batch_size = params['batch_size']

        logger=CSVLogger(['Epoch','Train Loss', 'Val MSE','ddg loss','dT loss'],f'logs/{fold}.csv')
        dl_train=DataLoader(ds_train, batch_size=batch_size,collate_fn=GraphCollate(),num_workers=args.workers,pin_memory=True,shuffle=True)
        dl_val=DataLoader(ds_val, batch_size=batch_size,collate_fn=GraphCollate(),num_workers=args.workers,pin_memory=True,shuffle=False)

        model, losses = train_model(exp_name, dl_train, dl_val, params, logger=logger, wandb_enabled=False, project=project)

        #torch.save(model.state_dict(),f'models/fold{fold}.pt')
        val_losses.append(losses)
        thermonet_models.append(model)

    d = pd.DataFrame([{k: np.min(v) for k, v in l.items()} for l in val_losses]).mean().to_dict()
#     with wandb.init(project=f'{project}-CV', name=name) as run:
#         run.log(d)
    return thermonet_models, d


params = copy.copy(BEST_PARAMS)
models = run_train(WANDB_TRAIN_NAME, params, project=WANDB_TRAIN_PROJECT)[0]
