# Install python
conda create -n humvis python=3.9

# Install pytorch
conda install pytorch==1.9.1 torchvision==0.10.1 -c pytorch

# Install PyTorch3D
conda install -c fvcore -c iopath -c conda-forge fvcore iopath

conda install -c bottler nvidiacub
# or
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0

pip install git+ssh://git@github.com/facebookresearch/pytorch3d.git@stable

# Install Flask
conda install flask

# Install other pkgs
pip install opencv-python
pip install scipy
pip install chumpy
# chumpy requires lower numpy version
pip install numpy==1.23.0
