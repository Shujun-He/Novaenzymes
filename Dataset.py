import torch
from torch.utils.data import Dataset
from biopandas.pdb import PandasPdb
import numpy as np

class e3nnDataset(Dataset):
    def __init__(self, df, features=None):
        self.df = df
        self.features = features
        self.atom_types = {'N':[1,0,0,0],
                           'C':[0,1,0,0],
                           'O':[0,0,1,0],
                           'S':[0,0,0,1]}

    def get_features(self,pdf_path,index,if_mutant,mutant_position=None,keep_radius=10):
        """
        Need:
        ``pos`` the position of the nodes (atoms)
        ``x`` the input features of the nodes, optional
        ``z`` the attributes of the nodes, for instance the atom type, optional
        ``batch`` the graph to which the node belong, optional

        """
        pdb=PandasPdb().read_pdb(pdf_path)
        pdb = pdb.df['ATOM'][pdb.df['ATOM']['element_symbol']!='H']

        #find atoms within radius
        pos=pdb[['x_coord','y_coord','z_coord']]
        pos_x=pdb['x_coord'].values
        pos_y=pdb['y_coord'].values
        pos_z=pdb['z_coord'].values
        pos=np.stack([pos_x,pos_y,pos_z],1)

        mutant_pos=pos[pdb['residue_number']==mutant_position]
        center=np.mean(mutant_pos,0)
        distance=((pos-center)**2).sum(-1)**0.5
        pdb=pdb[distance<keep_radius]

        #get features from atoms within radius
        pos=pdb[['x_coord','y_coord','z_coord']]
        pos_x=pdb['x_coord'].values
        pos_y=pdb['y_coord'].values
        pos_z=pdb['z_coord'].values
        pos=np.stack([pos_x,pos_y,pos_z],1)
        x=np.array([[*self.atom_types[a],if_mutant] for a in pdb['element_symbol']])
        batch=np.ones(len(pdb))*index


        pos, x, batch=torch.tensor(pos),torch.tensor(x),torch.tensor(batch)


        return pos, x, batch



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
        wt_pos,wt_x,wt_batch=self.get_features(wt_pdf_path,1,if_mutant=0,mutant_position=position)


        mt_pdf_path=f'../input/14656-unique-mutations-voxel-features-pdbs/pdbs/{pdb_name}/{pdb_name}_{wildtype}{position}{mutant}_relaxed.pdb'
        mt_pos,mt_x,mt_batch=self.get_features(mt_pdf_path,1,if_mutant=1,mutant_position=position)

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


class GraphCollate:
    def __init__(self):
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

            ddg.append(item['ddg'])
            ddt.append(item['ddt'])


        wt_pos=torch.cat(wt_pos)
        wt_x=torch.cat(wt_x)
        wt_batch=torch.cat(wt_batch)

        mt_pos=torch.cat(mt_pos)
        mt_x=torch.cat(mt_x)
        mt_batch=torch.cat(mt_batch)

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
