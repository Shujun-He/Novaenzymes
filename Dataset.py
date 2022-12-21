import torch
from torch.utils.data import Dataset
from biopandas.pdb import PandasPdb
import numpy as np
import pandas as pd

class e3nnDataset(Dataset):
    def __init__(self, df, features=None):
        self.df = df
        self.features = features
        self.atom_types = {'N':[1,0,0,0],
                           'C':[0,1,0,0],
                           'O':[0,0,1,0],
                           'S':[0,0,0,1]}
        aa_df=pd.read_csv("/home/exx/Documents/RNAplay/data/RNA_codons.csv")
        aa_types=aa_df['AminoAcid'].unique()

        self.aa_types={}
        cnt=0
        for aa in aa_types:
            if aa != 'Stp':
                one_hot=[0]*20
                one_hot[cnt]=1
                self.aa_types[aa.upper()]=one_hot
                cnt+=1

        # print(self.aa_types)
        # exit()


    def get_features(self,pdf_path,index,if_mutant,mutant_position=None,keep_radius=8):
        """
        Need:
        ``pos`` the position of the nodes (atoms)
        ``x`` the input features of the nodes, optional
        ``z`` the attributes of the nodes, for instance the atom type, optional
        ``batch`` the graph to which the node belong, optional

        """
        pdb=PandasPdb().read_pdb(pdf_path)

        # try:
        #     channels=np.load(pdf_path[:-3]+'channels.npy')
        #
        #     if len(channels)>len(pdb.df['ATOM']):
        #         channels=channels[:len(pdb.df['ATOM'])]
        #     elif len(channels)<len(pdb.df['ATOM']):
        #         channels=np.pad(channels,((0,len(pdb.df['ATOM'])-len(channels)),(0,0)))
        # except:
        #     channels=np.zeros((len(pdb.df['ATOM']),8))

        #channels=channels[pdb.df['ATOM']['element_symbol']!='H']
        pdb = pdb.df['ATOM'][pdb.df['ATOM']['element_symbol']!='H']


        # print(channels.shape)
        # exit()
        # print(pdb['residue_name'])
        # exit()
        #find atoms within radius
        pos=pdb[['x_coord','y_coord','z_coord']]
        pos_x=pdb['x_coord'].values
        pos_y=pdb['y_coord'].values
        pos_z=pdb['z_coord'].values
        pos=np.stack([pos_x,pos_y,pos_z],1)

        mutant_pos=pos[(pdb['residue_number']==mutant_position)*(pdb['element_symbol']=='C')]

        center=np.mean(mutant_pos,0)
        distance=((pos-center)**2).sum(-1)**0.5
        pdb=pdb[distance<keep_radius]

        #channels=channels[distance<keep_radius]


        #get features from atoms within radius
        pos=pdb[['x_coord','y_coord','z_coord']]
        pos_x=pdb['x_coord'].values
        pos_y=pdb['y_coord'].values
        pos_z=pdb['z_coord'].values
        pos=np.stack([pos_x,pos_y,pos_z],1)

        #aa_types=self.aa_types[aa]

        #x=np.array([[*self.atom_types[a],*self.aa_types[aa],*list(c),if_mutant] for a,c,aa in zip(pdb['element_symbol'],channels,pdb['residue_name'])])
        x=np.array([[*self.atom_types[a],*self.aa_types[aa],if_mutant] for a,aa in zip(pdb['element_symbol'],pdb['residue_name'])])
        # print(x)
        # exit()

        batch=np.ones(len(pdb))*index
        #batch[pdb['residue_number']==mutant_position]=index

        pos, x, batch, center=torch.tensor(pos),torch.tensor(x),torch.tensor(batch), torch.tensor(center)


        return pos, x, batch, center



    def __getitem__(self, item):
        r = self.df.iloc[item]
        pdb_name=r['PDB_chain']
        wildtype=r['wildtype']
        position=r['pdb_position']
        mutant=r['mutant']

        # r = self.df.iloc[item]
        # if 'ddG' in self.df.columns:
        #     return torch.as_tensor(r.features, dtype=torch.float), torch.tensor(r.ddG, dtype=torch.float), torch.tensor(r.dT, dtype=torch.float)
        # else:
        #     return torch.as_tensor(r.features, dtype=torch.float)

        wt_pdf_path=f'../input/14656-unique-mutations-voxel-features-pdbs/pdbs/{pdb_name}/{pdb_name}_relaxed.pdb'
        wt_pos,wt_x,wt_batch,wt_center=self.get_features(wt_pdf_path,1,if_mutant=0,mutant_position=position)


        mt_pdf_path=f'../input/14656-unique-mutations-voxel-features-pdbs/pdbs/{pdb_name}/{pdb_name}_{wildtype}{position}{mutant}_relaxed.pdb'
        mt_pos,mt_x,mt_batch,mt_center=self.get_features(mt_pdf_path,1,if_mutant=1,mutant_position=position)

        #mt_pos=mt_pos-mt_center+wt_center
        mt_pos=mt_pos-mt_center+wt_center+40

        ddg, ddt = torch.tensor(r.ddG, dtype=torch.float), torch.tensor(r.dT, dtype=torch.float)


        return {'wt_pos':wt_pos,
                'wt_x':wt_x,
                'wt_batch':wt_batch,
                'mt_pos':mt_pos,
                'mt_x':mt_x,
                'mt_batch':mt_batch,
                "ddg":ddg,
                "ddt":ddt}

    def __len__(self):
        return len(self.df) #if self.df is not None else len(self.features)

class e3nnDataset_test(e3nnDataset):
    def __init__(self, df, features=None):
        super().__init__(df=df)

        print(self.df)

    def __getitem__(self, item):
        r = self.df.iloc[item]
        #pdb_name=r['PDB_chain']
        #wildtype=r['wildtype']
        position=r['idx']
        mutant=r['mut'].replace('-','_')


        # r = self.df.iloc[item]
        # if 'ddG' in self.df.columns:
        #     return torch.as_tensor(r.features, dtype=torch.float), torch.tensor(r.ddG, dtype=torch.float), torch.tensor(r.dT, dtype=torch.float)
        # else:
        #     return torch.as_tensor(r.features, dtype=torch.float)

        wt_pdf_path=f'../input/novozymes-enzyme-stability-prediction/wildtype_structure_prediction_af2.pdb'
        wt_pos,wt_x,wt_batch,wt_center=self.get_features(wt_pdf_path,1,if_mutant=0,mutant_position=position)


        mt_pdf_path=f'../input/af_test_pdbs/{mutant}_unrelaxed_rank_1_model_3.pdb'
        if mutant=='0':
            mt_pdf_path=f'../input/novozymes-enzyme-stability-prediction/wildtype_structure_prediction_af2.pdb'
        mt_pos,mt_x,mt_batch,mt_center=self.get_features(mt_pdf_path,1,if_mutant=1,mutant_position=position)

        mt_pos=mt_pos-mt_center+wt_center+40
        #mt_pos=mt_pos+20

        #ddg, ddt = torch.tensor(r.ddG, dtype=torch.float), torch.tensor(r.dT, dtype=torch.float)


        return {'wt_pos':wt_pos,
                'wt_x':wt_x,
                'wt_batch':wt_batch,
                'mt_pos':mt_pos,
                'mt_x':mt_x,
                'mt_batch':mt_batch}


class GraphCollate:
    def __init__(self,test=False):
        self.test=test
        pass

    def __call__(self,data):
        wt_pos=[]
        wt_x=[]
        wt_batch=[]

        mt_pos=[]
        mt_x=[]
        mt_batch=[]


        ddg=[]
        ddt=[]
        for i,item in enumerate(data):
            wt_pos.append(item['wt_pos'])
            wt_x.append(item['wt_x'])
            wt_batch.append(item['wt_batch']*i)

            mt_pos.append(item['mt_pos'])
            mt_x.append(item['mt_x'])
            mt_batch.append(item['mt_batch']*i)

            if not self.test:
                ddg.append(item['ddg'])
                ddt.append(item['ddt'])


        wt_pos=torch.cat(wt_pos)
        wt_x=torch.cat(wt_x)
        wt_batch=torch.cat(wt_batch)

        mt_pos=torch.cat(mt_pos)
        mt_x=torch.cat(mt_x)
        mt_batch=torch.cat(mt_batch)

        if not self.test:
            ddg=torch.stack(ddg)
            ddt=torch.stack(ddt)

            return {'wt_pos':wt_pos,
                    'wt_x':wt_x,
                    'wt_batch':wt_batch,
                    'mt_pos':mt_pos,
                    'mt_x':mt_x,
                    'mt_batch':mt_batch,
                    "ddg":ddg,
                    "ddt":ddt}
        else:

            return {'wt_pos':wt_pos,
                    'wt_x':wt_x,
                    'wt_batch':wt_batch,
                    'mt_pos':mt_pos,
                    'mt_x':mt_x,
                    'mt_batch':mt_batch}
