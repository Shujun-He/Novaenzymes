from biopandas.pdb import PandasPdb


path='../input/14656-unique-mutations-voxel-features-pdbs/pdbs/1msiA/1msiA_relaxed.pdb'

mol = PandasPdb().read_pdb(path)
