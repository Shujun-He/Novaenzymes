git clone https://github.com/Shujun-He/Novaenzymes
mkdir input
kaggle kernels output shujun717/14656-unique-mutations-voxel-features-pdbs -p input/14656-unique-mutations-voxel-features-pdbs
unzip input/14656-unique-mutations-voxel-features-pdbs/pdbs.zip -d input/14656-unique-mutations-voxel-features-pdbs/
unzip input/14656-unique-mutations-voxel-features-pdbs/features.zip -d input/14656-unique-mutations-voxel-features-pdbs/
cp input/14656-unique-mutations-voxel-features-pdbs/dataset.csv input/14656-unique-mutations-voxel-features-pdbs/features
kaggle datasets download shujun717/rna-codons
unzip rna-codons.zip -d input/
