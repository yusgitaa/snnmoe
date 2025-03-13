conda create -n ljg_moe python==3.8.20
conda activate ljg_moe

pip install timm == 0.6.12
pip install spikingjelly
pip install torchinfo
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

module load cuda/11.3



srun -J jungang -p debug -N 1 -n 4 --cpus-per-task=4 --gres=gpu:1 --pty /bin/bash
module load cuda/11.3
conda activate ljg_moe
cd /hpc2hdd/JH_DATA/share/jsu360/PrivateShareGroup/jsu360_NLPGroup/lijungang/Research_Works/snnmoe
bash ./scripts/train/train_demo.sh