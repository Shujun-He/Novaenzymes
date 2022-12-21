import multiprocessing
import os

import Levenshtein
import numpy as np
from plotly.offline import init_notebook_mode
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import torch

init_notebook_mode(connected=True)
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
    parser.add_argument('--workers', type=int, default=16, help='number of workers for dataloader')
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
                fname = f'{MODELS_PATH}/{name}.pt'
                torch.save(model.state_dict(), fname)
            prog.set_description(
                f'Epoch: {epoch}; Train Loss: {train_loss:.02f} Val MSE:{eval_loss:.02f}; Min Val MSE:{min_loss:.02f}; ddg loss:{eval_ddg_loss:.02f}; dT loss:{eval_dt_loss:.02f}')
            logger.log([epoch,train_loss,eval_loss,eval_ddg_loss,eval_dt_loss])
            if epoch - min_epoch > EARLY_STOPPING_PATIENCE:
                print('Early stopping')
                break

    if run is not None:
        #art = wandb.Artifact("thermonet2", type="model")
        fname = f'{MODELS_PATH}/{name}.pt'
        torch.save(model.state_dict(), fname)
        art.add_file(fname)
        run.log_artifact(art)
        run.finish()
    return best_model, val_losses


def run_train(name, params, project='thermonetv2'):
    os.makedirs(MODELS_PATH, exist_ok=True)
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
        fname = f'{MODELS_PATH}/{exp_name}.pt'
        # ds_train = ThermoNet2Dataset(df_train.loc[train_idx])
        # ds_val = ThermoNet2Dataset(df_train.loc[val_idx])

        ds_train = e3nnDataset(df_train.loc[train_idx])
        ds_val = e3nnDataset(df_train.loc[val_idx])

        batch_size = params['batch_size']

        logger=CSVLogger(['Epoch','Train Loss', 'Val MSE','ddg loss','dT loss'],f'logs/{fold}.csv')
        dl_train=DataLoader(ds_train, batch_size=batch_size,collate_fn=GraphCollate(),num_workers=args.workers,pin_memory=True,shuffle=True)
        dl_val=DataLoader(ds_val, batch_size=batch_size,collate_fn=GraphCollate(),num_workers=args.workers,pin_memory=True,shuffle=False)

        model, losses = train_model(exp_name, dl_train, dl_val, params, logger=logger, wandb_enabled=False, project=project)

        torch.save(model.state_dict(),f'models/fold{fold}.pt')
        val_losses.append(losses)
        thermonet_models.append(model)

    d = pd.DataFrame([{k: np.min(v) for k, v in l.items()} for l in val_losses]).mean().to_dict()
#     with wandb.init(project=f'{project}-CV', name=name) as run:
#         run.log(d)
    return thermonet_models, d


if TRAIN:
    params = copy.copy(BEST_PARAMS)
    thermonet_models = run_train(WANDB_TRAIN_NAME, params, project=WANDB_TRAIN_PROJECT)[0]




def gen_mutations(name, df,
                  wild="VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQ""RVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGT""NAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKAL""GSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK"):
    result = []
    for _, r in df.iterrows():
        ops = Levenshtein.editops(wild, r.protein_sequence)
        assert len(ops) <= 1
        if len(ops) > 0 and ops[0][0] == 'replace':
            idx = ops[0][1]
            result.append([ops[0][0], idx + 1, wild[idx], r.protein_sequence[idx]])
        elif len(ops) == 0:
            result.append(['same', 0, '', ''])
        elif ops[0][0] == 'insert':
            assert False, "Ups"
        elif ops[0][0] == 'delete':
            idx = ops[0][1]
            result.append(['delete', idx + 1, wild[idx], '-'])
        else:
            assert False, "Ups"

    df = pd.concat([df, pd.DataFrame(data=result, columns=['op', 'idx', 'wild', 'mutant'])], axis=1)
    df['mut'] = df[['wild', 'idx', 'mutant']].astype(str).apply(lambda v: ''.join(v), axis=1)
    df['name'] = name
    return df

if SUBMISSION:
    df_test = gen_mutations('wildtypeA', pd.read_csv(TEST_CSV))
    #display(df_test)


# In[118]:


def predict(model:ThermoNet2, test_features):
    with torch.no_grad():
        model.eval()
        dl = DataLoader(ThermoNet2Dataset(features=test_features), batch_size=64)
        if IS_DDG_TARGET:
            return np.concatenate(
                [model.forward(x.to(DEVICE))[0].cpu().numpy() for x in tqdm(dl, desc='ThermoNet2 ddg predict', disable=True)])
        else:
            return np.concatenate(
                [model.forward(x.to(DEVICE))[1].cpu().numpy() for x in tqdm(dl, desc='ThermoNet2 dt predict', disable=True)])


# In[121]:


if SUBMISSION:
    #thermonet_models = [load_pytorch_model(f) for f in tqdm(glob.glob(f'artifacts/*/{WANDB_TRAIN_NAME}*.pt'), desc=f'Loading models {WANDB_TRAIN_NAME}')]

    test_features = np.load(TEST_FEATURES_PATH)
    test_ddg = np.stack([predict(model, test_features) for model in tqdm(thermonet_models, desc='Fold prediction')])
    test_ddg = np.mean(test_ddg, axis=0).flatten()

    # replacement mutations
    df_test.loc[df_test.op == 'replace', 'ddg'] = test_ddg
    # deletion mutations
    df_test.loc[df_test['op'] == "delete", 'ddg'] = df_test[df_test["op"]=="replace"]["ddg"].quantile(q=0.25)
    # no mutations
    df_test.loc[df_test['op'] == "same", 'ddg'] = 0.

    df_test.rename(columns={'ddg': 'tm'})[['seq_id', 'tm']].to_csv('submission.csv', index=False)
    #get_ipython().system('head submission.csv')


# # Ensemble
#
# Ensembling ThermoNetV2 with top public solutions

# In[123]:


if SUBMISSION:

    def ranked(f):
        return rankdata(pd.read_csv(f).tm)

    pred = 0.7 * ranked('../input/rmsd-from-molecular-dynamics/submission_rmsd.csv')+\
        0.3 * (ranked('../input/plldt-ddg-demask-sasa/deepddg-ddg.csv')+        \
        ranked('../input/novo-esp-eli5-performant-approaches-lb-0-451/submission.csv')+ \
        ranked('../input/nesp-alphafold-getarea-exploration/submission.csv') + \
        ranked('submission.csv'))


    df = pd.read_csv('../input/novozymes-enzyme-stability-prediction/sample_submission.csv')
    df.tm = pred


    # equally weighted ensemble with https://www.kaggle.com/code/shlomoron/nesp-relaxed-rosetta-scores
    df.tm = rankdata(df.tm) + ranked('../input/nesp-relaxed-rosetta-scores/submission_rosetta_scores')


    df.to_csv('ensemble_submission.csv', index=False)
    #get_ipython().system('head ensemble_submission.csv')


# In[ ]:


#get_ipython().system('rm -rf wandb')


#
# <div class="alert alert-block alert-danger" style="text-align:center; font-size:20px;">
#     ❤️ Dont forget to ▲upvote▲ if you find this notebook usefull!  ❤️
# </div>
#
