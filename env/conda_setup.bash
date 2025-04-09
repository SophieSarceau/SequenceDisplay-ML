#! /bin/bash

conda create --name sdeml python=3.10

conda activate sdeml

# Install the required packages -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy==1.26.4
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install pytorch_lightning==2.2.4
pip install wandb==0.17.0
pip install pandas==2.2.2
pip install scikit-learn==1.4.2
pip install scipy==1.13.1
pip install tqdm==4.64.0 
pip install matplotlib==3.8.0
pip install fair-esm==2.0.0
pip install omegaconf==2.3.0
