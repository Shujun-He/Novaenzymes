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

from Network import ThermoNet2


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
 'batch_size': 256,
 'AdamW_decay': 1.3994535042337082,
 'dropout_rate': 0.06297340526648805,
 'learning_rate': 0.00020503764745082723,
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

df_train = load_data()
df_train


# In[101]:


df_train.dT.plot.hist(title='Distribution of dT')


# In[102]:


df_train.ddG.plot.hist(title='Distribution of ddG')


# In[103]:


df_train.plot.scatter(x='ddG', y='dT', title='ddG vs dT')


# In[104]:


df_train.groupby('sequence').features.count().plot.hist(title='Mutations per sequence', bins=50)


# # Plotting voxel representation of features
#
# In the following plots we use 3D scatterplot to demonstrate training samples.
# Specifically we plot `occupancy` feature that represents probability that certain voxel is occpupied by an atom.
# Recall that each training/test sample uses a combination of wildetype+mutant features. So we use the following color-coding:
# * blue color represents voxels that are occupied in both wildtype and mutant structures
# * red color represents voxels that are occupied only in the mutant structure
# * green color represents voxels that are occupied only in the wildtype structure

# In[105]:


from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

def plot_voxels():
    for i in [123, 124, 125, 126]:
        df = pd.DataFrame([(x, y1, z) for x in range(16) for y1 in range(16) for z in range(16)], columns=['x', 'y', 'z'])
        df['occupancy1'] = df_train.iloc[i].features[6, :, :, :].flatten() > 0.9
        df['occupancy2'] = df_train.iloc[i].features[13, :, :, :].flatten() > 0.9
        df.loc[df.occupancy1 | df.occupancy2, 'color'] = 'blue'
        df.loc[~df.occupancy1 & df.occupancy2, 'color'] = 'red'
        df.loc[df.occupancy1 & ~df.occupancy2, 'color'] = 'green'
        ddg = df_train.ddG[i]
        fig = px.scatter_3d(df.dropna(), x='x', y='y', z='z', color='color', title=f"Train idx:{i}; ddg={ddg}")
        fig.show()


plot_voxels()


# # Model

# In[106]:




if DEBUG:
    params = copy.copy(BEST_PARAMS)
    params['diff_features'] = False
    tn2 =ThermoNet2(params)
    print([out.shape for out in tn2.forward(torch.randn((2, 14, 16, 16, 16)))])
    print(tn2)


# # Dataset

# In[107]:


class ThermoNet2Dataset(Dataset):
    def __init__(self, df=None, features=None):
        self.df = df
        self.features = features

    def __getitem__(self, item):
        if self.df is not None:
            r = self.df.iloc[item]
            if 'ddG' in self.df.columns:
                return torch.as_tensor(r.features, dtype=torch.float), torch.tensor(r.ddG, dtype=torch.float), torch.tensor(r.dT, dtype=torch.float)
            else:
                return torch.as_tensor(r.features, dtype=torch.float)
        else:
            return torch.as_tensor(self.features[item], dtype=torch.float)

    def __len__(self):
        return len(self.df) if self.df is not None else len(self.features)

if DEBUG:
    ds = ThermoNet2Dataset(df_train)
    feat, t1, t2 = next(iter(DataLoader(ds, batch_size=BEST_PARAMS['batch_size'])))
    print(feat.shape, t1.shape, t2.shape)


# In[108]:


df_train.head()


# # Train

# In[109]:


def evaluate(model, dl_val, params):
    criterion = nn.MSELoss()
    model.eval()
    losses = []
    ddg_preds = []
    dt_preds = []
    ddg_losses = []
    dt_losses = []
    with torch.no_grad():
        for x, ddg, dt in tqdm(dl_val, desc='Eval', disable=True):
            ddg_pred, dt_pred = model(x.to(DEVICE))
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


def load_pytorch_model(fname, params=BEST_PARAMS):
    model = ThermoNet2(params)
    model.load_state_dict(torch.load(fname))
    return model


def train_model(name, dl_train, dl_val, params, logger, wandb_enabled=False, project='thermonetv2'):
    model = ThermoNet2(params).to(DEVICE)

    if params['AdamW']:
        def get_optimizer_params(model, encoder_lr, weight_decay=0.0):
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'lr': encoder_lr, 'weight_decay': weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'lr': encoder_lr, 'weight_decay': 0.0},
            ]
            return optimizer_parameters

        optimizer_parameters = get_optimizer_params(model,
                                                    encoder_lr=params['learning_rate'],
                                                    weight_decay=params['AdamW_decay'])
        optim = AdamW(optimizer_parameters, lr=params['learning_rate'])
    else:
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
            for x, ddg, dt in tqdm(dl_train, desc='Train', disable=True):

                ddg_pred, dt_pred = model(x.to(DEVICE))
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

            train_loss=np.mean(train_loss)
            eval_loss, eval_ddg_loss, eval_dt_loss = evaluate(model, dl_val, params)[:3]
            val_losses['loss'].append(eval_loss)
            val_losses['ddg_loss'].append(eval_ddg_loss)
            val_losses['dt_loss'].append(eval_dt_loss)
            if run is not None:
                run.log({'val_loss': eval_loss, 'val_ddg_loss': eval_ddg_loss, 'val_dt_loss': eval_dt_loss,
                         'lr': scheduler.get_last_lr()[0] if scheduler is not None else params['learning_rate']})
            if eval_loss < min_loss:
                min_loss = eval_loss
                min_epoch = epoch
                best_model = copy.deepcopy(model)
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
    kfold = GroupKFold(N_FOLDS)
    if params['GroupKFold']:
        groups = df_train.sequence
    else:
        groups = range(len(df_train))
    for fold, (train_idx, val_idx) in enumerate(
            tqdm(kfold.split(df_train, groups=groups), total=N_FOLDS, desc="Folds")):
        exp_name = f'{name}-{fold}'
        fname = f'{MODELS_PATH}/{exp_name}.pt'
        ds_train = ThermoNet2Dataset(df_train.loc[train_idx])
        ds_val = ThermoNet2Dataset(df_train.loc[val_idx])

        batch_size = params['batch_size']

        logger=CSVLogger(['Epoch','Train Loss', 'Val MSE','ddg loss','dT loss'],f'logs/{fold}.csv')
        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
        dl_val = DataLoader(ds_val, batch_size=64, pin_memory=True, drop_last=True)

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


# In[110]:


#%wandb -h 800 vslaykovsky/ThermoNetV2-train


# # Optuna Hyperparameter Optimization

# In[111]:


if OPTUNA:

    import optuna
    from optuna.integration.wandb import WeightsAndBiasesCallback

    wandbc = WeightsAndBiasesCallback(wandb_kwargs={"project": OPTUNA_WANDB_PROJECT}, as_multirun=True)

    @wandbc.track_in_wandb()
    def objective(trial):
        params = copy.copy(DEFAULT_PARAMS)
        params['conv_layer_num'] = trial.suggest_int('conv_layer_num', 3, 6)
        params['AdamW'] = trial.suggest_categorical('AdamW', [True, False])
        if params['AdamW']:
            params['AdamW_decay'] = trial.suggest_float('AdamW_decay', 0.001, 100, log=True)
        params['dense_layer_size'] = trial.suggest_int('dense_layer_size', 16, 128, log=True)

        params['dropout_rate'] = trial.suggest_float('dropout_rate', 0., 0.7, log=False)
        params['dropout_rate_dt'] = trial.suggest_float('dropout_rate_dt', 0., 0.7, log=False)
        params['learning_rate'] = trial.suggest_float('learning_rate', 5e-6, 1e-3, log=True)
        params['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
        params['C_dt_loss'] = trial.suggest_categorical('C_dt_loss', [0, 0.01, 0.1, 1.])
        params['GroupKFold'] = True # works best for hyperparameter optimization
        params['OneCycleLR'] = trial.suggest_categorical('OneCycleLR', [True, False])

        print('params', params)
        # --------------- train --------------
        kfold = GroupKFold(5)
        for train_idx, val_idx in kfold.split(df_train, groups=df_train.sequence):
            ds_train = ThermoNet2Dataset(df_train.loc[train_idx])
            ds_val = ThermoNet2Dataset(df_train.loc[val_idx])
            batch_size = params['batch_size']
            dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
            dl_val = DataLoader(ds_val, batch_size=64, pin_memory=True, shuffle=True, drop_last=True)
            _, losses = train_model("optuna", dl_train, dl_val, params, wandb_enabled=False)
            return np.min(losses['ddg_loss' if IS_DDG_TARGET else 'dt_loss'])


    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS, callbacks=[wandbc])


# In[112]:


#%wandb -h 1200 vslaykovsky/ThermoNetV2-Optuna-fireprotdb


# # Wandb Sweeps Hyperparameter Optimization
#
# Wandb sweeps are executed in 2 steps:
# 1. Create sweep configuration in your Wandb project. This is done with `wandb.sweep` call below.
# 2. Once you got your sweep id, pass it to your agents. Agents run tests using configuration passed from Wandb servers using `wandb.agent` call.
#
# See more info on Wandb sweeps here https://docs.wandb.ai/guides/sweeps

# In[113]:


# Set your sweep_id below to start optimization
#%env SWEEP_ID=xxxxx


# In[114]:


if WANDB_SWEEP:
    import wandb
    sweep_id = os.environ.get('SWEEP_ID')
    print('wandb sweep ', sweep_id)

    if sweep_id is None:
        """
        First run. Generate sweep_id.
        """
        sweep_id = wandb.sweep(sweep={
            'method': 'bayes',
            'name': 'thermonet2-sweep',
            'metric': {'goal': 'minimize', 'name': 'val_ddg_loss'},
            'parameters':
                {
                    'conv_layer_num': {'values': [3, 4, 5, 6]},
                    'AdamW': {'values': [True, False]},
                    'AdamW_decay': {'min': 0.01, 'max': 5, 'distribution': 'log_uniform_values'},
                    'dense_layer_size': {'min': 16, 'max': 128, 'distribution': 'log_uniform_values'},
                    'dropout_rate': {'min': 0., 'max': 0.8, 'distribution': 'uniform'},
                    'dropout_rate_dt': {'min': 0., 'max': 0.8, 'distribution': 'uniform'},
                    'learning_rate': {'min': 1e-5, 'max': 3e-3, 'distribution': 'log_uniform_values'},
                    'batch_size': {'values': [64, 128, 256, 512]},
                    'OneCycleLR': {'values': [True, False]},
                    'C_dt_loss': {'values': [0, 0.003, 0.01, 0.03]},
                }
        }, project=WANDB_SWEEP_PROJECT)
        print('Generated sweep id', sweep_id)
    else:
        """
        Agent run. Use sweep_id generated above.
        """
        def wandb_callback():
            with wandb.init() as run:
                params = copy.copy(DEFAULT_PARAMS)
                params['conv_layer_num'] = run.config.conv_layer_num
                params['AdamW'] = run.config.AdamW
                params['AdamW_decay'] = run.config.AdamW_decay
                params['dense_layer_size'] = int(run.config.dense_layer_size)
                params['dropout_rate'] = run.config.dropout_rate
                params['dropout_rate_dt'] = run.config.dropout_rate_dt
                params['learning_rate'] = run.config.learning_rate
                params['batch_size'] = run.config.batch_size
                params['OneCycleLR'] = run.config.OneCycleLR
                params['C_dt_loss'] = run.config.C_dt_loss
                params['GroupKFold'] = True  # only for hyperparameter optimization
                print('params', params)

                # --------------- train --------------
                kfold = GroupKFold(5)
                for train_idx, val_idx in kfold.split(df_train, groups=df_train.sequence):
                    ds_train = ThermoNet2Dataset(df_train.loc[train_idx])
                    ds_val = ThermoNet2Dataset(df_train.loc[val_idx])
                    batch_size = params['batch_size']
                    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, pin_memory=True)
                    dl_val = DataLoader(ds_val, batch_size=512, pin_memory=True, shuffle=True)
                    _, losses = train_model(0, dl_train, dl_val, params, wandb_enabled=False)
                    for epoch in range(len(losses['ddg_loss'])):
                        run.log({
                            'epoch': epoch,
                            'val_loss': losses['loss'][epoch],
                            'val_ddg_loss': losses['ddg_loss'][epoch],
                            'val_dt_loss': losses['dt_loss'][epoch],
                        })
                    break



        # Start sweep job.
        wandb.agent(sweep_id, project=WANDB_SWEEP_PROJECT, function=wandb_callback, count=100000)


# In[115]:


#%wandb -h 1200 vslaykovsky/ThermoNetV2-fireprot-sweep/sweeps/vzyvxo1a


# # Submission
#
# All models are stored in Wandb, so downloading models to the localhost here.

# In[116]:


# def collect_wandb_models(name):
#     runs = wandb.Api().runs(
#         path='vslaykovsky/ThermoNetV2-train',
#     )
#     with tqdm(runs, desc='Downloading artefacts') as prog:
#         for run in prog:
#             art = run.logged_artifacts()
#             if len(art) > 0:
#                 if name in run.name:
#                     prog.set_description(run.name)
#                     art[0].download()


# if SUBMISSION:
#     collect_wandb_models(WANDB_TRAIN_NAME)


# In[117]:


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
