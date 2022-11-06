# Get docker running on ubuntu
https://docs.docker.com/engine/install/ubuntu/#set-up-the-repository

# Get Nvidia Container Toolkit
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

# Get pytorch docker
docker pull pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Run it to confirm
sudo docker run --gpus=all --rm -it --entrypoint bash pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Build Images
## Pytorch
Windows:
cd C:\Users\rober\Dropbox\projects\fun\deep_learning\gan_pipeline\docker\gan_pipeline & docker build -f DockerfileTorch -t rluedeman/gan-pipeline .
Linux:
cd ~/DeepLearning/gan_pipeline/docker/gan_pipeline
sudo docker build -f DockerfileTorch -t rluedeman/gan-pipeline .

## Tensorflow
cd C:\Users\rober\Dropbox\projects\fun\deep_learning\docker\gan_pipeline
sudo docker build -f DockerfileTensorflow -t rluedeman/gan-pipeline-tensorflow .

# Run the new image
Windows:
docker run --rm -it --entrypoint bash rluedeman/gan-pipeline
Ubuntu:
sudo docker run --gpus=all --rm -it --entrypoint bash rluedeman/gan-pipeline
# With mounted files
Windows:
docker run --rm -it --mount type=bind,source=C:\Users\rober\Dropbox\projects\fun\deep_learning,target=/src -p 5000:5000 --entrypoint bash rluedeman/gan-pipeline

# Test basic functionality:
> cd /src/; uvicorn gan_pipeline.app.gan_pipeline_api:app --host 0.0.0.0 --port 5000 --reload 

# Run using compose
## Bash
cd ~/DeepLearning/gan_pipeline/docker/gan_pipeline/compose; sudo docker-compose -f shell-linux.yml run bash_shell
import torch; print(torch.cuda.is_available())

# Tensorflow
cd ~/DeepLearning/docker/learning_deep/compose; sudo docker-compose -f shell-tensorflow.yml run bash_shell
ipython
import tensorflow as tf; print(tf.test.is_gpu_available())

# Run Bash with Compose
Windows:
cd C:\Users\rober\Dropbox\projects\fun\deep_learning\gan_pipeline\docker\gan_pipeline\compose & docker-compose -f shell-win.yml run bash_shell
Linux:
cd ~/DeepLearning/gan_pipeline/docker/gan_pipeline/compose; sudo docker-compose -f shell-linux.yml run bash_shell

# Run the APIs
Windows:
cd C:\Users\rober\Dropbox\projects\fun\deep_learning\gan_pipeline\docker\gan_pipeline\compose & docker-compose -f api-win.yml up

# Docker utils:
# remove everything
sudo docker system prune -a --volumes
# Kill all containers
sudo docker kill $(sudo docker ps -q)


# Dataset generation
python dataset_tool.py --resolution=512x512 --source=../../datasets/gan_pipeline/Tractors/Data/Training/ --dest=../../datasets/gan_pipeline/Tractors/Data/tractors512.zip

python dataset_tool.py --resolution=64x64 --source=/desktop/gan_pipeline/foundation/Data/Training/ --dest=/desktop/gan_pipeline/foundation/Data/foundation_64_400k.zip

python dataset_tool.py --resolution=128x128 --source=/desktop/gan_pipeline/foundation/Data/Training/ --dest=/desktop/gan_pipeline/foundation/Data/foundation_128_400k.zip

python dataset_tool.py --resolution=256x256 --source=/desktop/gan_pipeline/foundation/Data/Training/ --dest=/desktop/gan_pipeline/foundation/Data/foundation_256_400k.zip


# Run stylegan
cd /src/sandbox/stylegan3
python train.py --cfg=stylegan2 --gamma=0.8192 --gpus=2 --batch=64 --batch-gpu=32 --mirror=1 --snap=20 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384 --kimg=30000 --data=../../datasets/gan_pipeline/Tractors/Data/tractors512_50k.zip --outdir ../learning_deep/experiments/results/tractors_512/
1
python train.py --cfg=stylegan2 --gamma=0.4096 --gpus=2 --batch=64 --batch-gpu=32 --mirror=1 --snap=20 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384 --kimg=30000 --data=../../datasets/gan_pipeline/Food/Data/Food512_80k.zip --outdir=../../datasets/gan_pipeline/Food/experiments

# Resume stylegan
cd /src/sandbox/stylegan3
python train.py --cfg=stylegan2 --outdir ../learning_deep/experiments/results/tractors_512/ --data=../../datasets/gan_pipeline/Tractors/Data/tractors512_50k.zip --gamma=0.4096 --gpus=2 --batch=64 --batch-gpu=32 --mirror=1 --snap=20 --map-depth=2 --glr=0.00125 --dlr=0.00125 --cbase=16384 --kimg=30000 --resume=../learning_deep/experiments/results/tractors_512/00011-stylegan2-tractors512_50k-gpus2-batch64-gamma0.4096/network-snapshot-002741.pkl

# Slow
python train.py --cfg=stylegan2 --outdir ../learning_deep/experiments/results/tractors_512/ --data=../../datasets/gan_pipeline/Tractors/Data/tractors512_50k.zip --gamma=0.4096 --gpus=2 --batch=64 --batch-gpu=32 --mirror=1 --snap=20 --map-depth=2 --glr=0.00125 --dlr=0.00125 --cbase=16384 --kimg=30000 --resume=../learning_deep/experiments/results/tractors_512/00011-stylegan2-tractors512_50k-gpus2-batch64-gamma0.4096/network-snapshot-002741.pkl

# Fast
python train.py --cfg=stylegan2 --outdir ../learning_deep/experiments/results/tractors_512/ --data=../../datasets/gan_pipeline/Tractors/Data/tractors512_50k.zip --gamma=0.8192 --gpus=2 --batch=64 --batch-gpu=32 --mirror=1 --snap=20 --map-depth=2 --glr=0.0005 --dlr=0.0005 --cbase=16384 --kimg=30000 --resume=../learning_deep/experiments/results/tractors_512/00017-stylegan2-tractors512_50k-gpus2-batch64-gamma0.8192/network-snapshot-000080.pkl



# Train at 128 to start
python train.py --cfg=stylegan2  --gamma=0.2 --gpus=2 --batch=128 --batch-gpu=64 --mirror=1 --tick=90 --snap=5 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=8192 --kimg=30000 --outdir ../../datasets/gan_pipeline/Food/experiments --data=../../datasets/gan_pipeline/Food/Data/Food128_96k.zip

python train.py --cfg=stylegan2  --gamma=0.2 --gpus=2 --batch=128 --batch-gpu=64 --mirror=1 --tick=90 --snap=5 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384 --kimg=30000 --outdir ../../datasets/gan_pipeline/Food/experiments --data=../../datasets/gan_pipeline/Food/Data/Food128_160k.zip

# Train at 256
python train.py --cfg=stylegan2  --gamma=1.6 --gpus=2 --batch=128 --batch-gpu=64 --mirror=1 --tick=45 --snap=5 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384 --kimg=30000 --outdir ../../datasets/gan_pipeline/Space/experiments --data=../../datasets/gan_pipeline/Food/Space/space256_50k.zip

python train.py --cfg=stylegan2  --gamma=1.6 --gpus=2 --batch=192 --batch-gpu=48 --mirror=1 --tick=25 --snap=4 --map-depth=2 --glr=0.0005 --dlr=0.0005 --cbase=32768 --kimg=20000 --outdir ../../datasets/gan_pipeline/Food/experiments --data=../../datasets/gan_pipeline/Food/Data/Food256_160k.zip --resume=../../datasets/gan_pipeline/Food/experiments/00037-stylegan2-Food256_160k-gpus2-batch128-gamma1.6/network-snapshot-001351.pkl

# Train at 512
python train.py --cfg=stylegan2  --gamma=1.6 --gpus=2 --batch=128 --batch-gpu=64 --mirror=1 --tick=45 --snap=5 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384 --kimg=30000 --outdir ../../datasets/gan_pipeline/Space/experiments --data=../../datasets/gan_pipeline/Space/Data/space512_50k.zip --resume=../../datasets/


cd /src/sandbox/stylegan3
python train.py --cfg=stylegan2  --gamma=0.2 --gpus=2 --batch=128 --batch-gpu=64 --mirror=1 --tick=90 --snap=5 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=8192 --kimg=30000 --outdir ../../datasets/gan_pipeline/Space/experimenta --data=../../datasets/gan_pipeline/Space/Data/space128_50k.zip



# Big Batch Size:
python train.py --cfg=stylegan2  --gamma=0.2 --gpus=2 --batch=512 --batch-gpu=64 --mirror=1 --tick=90 --snap=5 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=8192 --kimg=30000 --outdir=/data/DeepLearning/GanPipeline/Pasta/Experiments/ --data=/data/DeepLearning/GanPipeline/Pasta/Data/Pasta128_70k.zip

python train.py --cfg=stylegan2  --gamma=0.2 --gpus=2 --batch=1024 --batch-gpu=64 --mirror=1 --tick=90 --snap=5 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=8192 --kimg=30000 --outdir=/data/DeepLearning/GanPipeline/Pasta/Experiments/ --data=/data/DeepLearning/GanPipeline/Pasta/Data/Pasta128_70k.zip --resume=

python train.py --cfg=stylegan2  --gamma=0.2 --gpus=2 --batch=1920 --batch-gpu=96 --mirror=1 --tick=90 --snap=5 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=8192 --kimg=30000 --outdir=/data/DeepLearning/GanPipeline/Pasta/Experiments/ --data=/data/DeepLearning/GanPipeline/Pasta/Data/Pasta128_70k.zip --resume=



# Foundation Training
python train.py --cfg=stylegan2  --gamma=0.2 --gpus=2 --batch=1920 --batch-gpu=96 --mirror=1 --tick=90 --snap=5 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=8192 --kimg=50000 --outdir=/data/DeepLearning/GanPipeline/Foundation/Experiments/ --data=/data/DeepLearning/GanPipeline/Foundation/Data/foundation_128_200k.zip --resume=/data/DeepLearning/GanPipeline/Foundation/Experiments/00001-stylegan2-foundation_128_200k-gpus2-batch1920-gamma0.2/network-snapshot-003160.pkl