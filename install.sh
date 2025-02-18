# Assuming cuda is 12.x

conda create -n mvdp python=3.12 -y
conda activate mvdp

pip install -r requirements.txt

# installing pytorch3d: conda install directly often fails for version checking :( so installing from tar is much easier.
wget https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py312_cu121_pyt241.tar.bz2
conda install pytorch3d-0.7.8-py312_cu121_pyt241.tar.bz2
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia