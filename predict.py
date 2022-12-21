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

MULTIPROCESSING = False
BOXSIZE = 16
VOXELSIZE = 1
N_FOLDS = 10
MODELS_PATH = 'models'
DEBUG = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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


DEFAULT_PARAMS = {
    'SiLU': False,
    'diff_features': True,
    'LayerNorm': False,
    'GroupKFold': False,  # only use for hyperopt
    'epochs': 30,
    'AdamW': False,
}





BEST_PARAMS = {**DEFAULT_PARAMS, **{'AdamW': True,
 'C_dt_loss': 0.01,
 'OneCycleLR': False,
 'batch_size': 16,
 'AdamW_decay': 1.3994535042337082,
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


# test=np.load(TEST_FEATURES_PATH)
#
# exit()
# except Exception as ex:
#     print('Running locally')
#     WILDTYPE_PDB = 'nesp/thermonet/wildtypeA.pdb'
#     PDB_PATH = 'nesp/thermonet/'
#     TRAIN_FEATURES_PATH = 'data/train_features/features.npy'
#     TRAIN_TARGETS_PATH = 'data/train_features/dataset.csv'
#     TEST_FEATURES_PATH = 'data/nesp/nesp_features.npy'
#     TEST_CSV = 'data/nesp/test.csv'
#     PUBLIC_SUBMISSIONS=glob.glob('data/nesp/public_submissions/*.csv')
#     TRAIN_FEATURES_DIR = 'data/train_features/'
#     WANDB_API_KEY='your_key_here'

os.makedirs(MODELS_PATH, exist_ok=True)


# In[99]:


# import wandb

# """
# Add WANDB_API_KEY with your wandb.ai API key to run the code.
# """
# wandb.login(key=WANDB_API_KEY)


# # Load training data

# In[100]:


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

    df.features = [np.load(f) for f in tqdm(df.features, desc="2. Loading features")]


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

#df_train = load_data()
#df_train = pd.read_csv(f'{TRAIN_FEATURES_DIR}/dataset.csv')
#df_train


if DEBUG:
    params = copy.copy(BEST_PARAMS)
    params['diff_features'] = False
    tn2 =ThermoNet2(params)
    print([out.shape for out in tn2.forward(torch.randn((2, 14, 16, 16, 16)))])
    print(tn2)


# # Dataset

# In[107]:


#exit()















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



test_dataset=e3nnDataset_test(df_test[df_test.op == 'replace'])
test_loader = DataLoader(test_dataset, batch_size=32,collate_fn=GraphCollate(test=True),num_workers=32)

models=[]
for i in range(8):
    model=e3nnNetwork().double()
    model.eval()
    model.load_state_dict(torch.load(f'models/thermonetv2-7633-v2-{i}.pt'))
    model=model.to(DEVICE)
    models.append(model)


preds=[]
with torch.no_grad():
    for batch in tqdm(test_loader):

        batch_preds=[]
        for model in models:
            for key in batch:
                batch[key]=batch[key].to(DEVICE)
            ddg_pred, dt_pred = model(batch)
            batch_preds.append(ddg_pred)
        batch_preds=torch.stack(batch_preds,0).mean(0)
        preds.append(batch_preds.cpu())

test_ddg=torch.cat(preds).numpy()

#exit()
# exit()
#
# # In[118]:
#
#
# def predict(model:ThermoNet2, test_features):
#     with torch.no_grad():
#         model.eval()
#         dl = DataLoader(ThermoNet2Dataset(features=test_features), batch_size=64)
#         if IS_DDG_TARGET:
#             return np.concatenate(
#                 [model.forward(x.to(DEVICE))[0].cpu().numpy() for x in tqdm(dl, desc='ThermoNet2 ddg predict', disable=True)])
#         else:
#             return np.concatenate(
#                 [model.forward(x.to(DEVICE))[1].cpu().numpy() for x in tqdm(dl, desc='ThermoNet2 dt predict', disable=True)])


# In[121]:


if SUBMISSION:
    #thermonet_models = [load_pytorch_model(f) for f in tqdm(glob.glob(f'artifacts/*/{WANDB_TRAIN_NAME}*.pt'), desc=f'Loading models {WANDB_TRAIN_NAME}')]

    #test_features = np.load(TEST_FEATURES_PATH)
    #test_ddg = np.stack([predict(model, test_features) for model in tqdm(thermonet_models, desc='Fold prediction')])
    #test_ddg = np.mean(test_ddg, axis=0).flatten()

    #df_test.loc[:, 'ddg'] = test_ddg
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
