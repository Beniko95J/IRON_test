conda create -y -n iron python=3.8 && conda activate iron
pip install numpy scipy==1.9 trimesh opencv_python scikit-image imageio imageio-ffmpeg pyhocon==0.3.59 PyMCubes tqdm icecream configargparse
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install tensorboard kornia
conda install -c conda-forge igl
