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

# Run stylegan
cd /src/sandbox/stylegan3
python train.py --cfg=stylegan2 --outdir ../learning_deep/experiments/results/tractors_512/ --data=../../datasets/gan_pipeline/Tractors/Data/tractors512.zip --gamma=0.4096 --gpus=2 --batch=64 --batch-gpu=32 --mirror=1 --snap=20 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384 --kimg=30000

# Resume stylegan
cd /src/sandbox/stylegan3
python train.py --cfg=stylegan2 --outdir ../learning_deep/experiments/results/tractors_512/ --data=../../datasets/gan_pipeline/Tractors/Data/tractors512.zip --resume=../learning_deep/experiments/results/tractors_512/00000-stylegan2-tractors512-gpus2-batch64-gamma0.4096/network-snapshot-003145.pkl --gamma=0.4096 --gpus=2 --batch=64 --batch-gpu=32 --mirror=1 --snap=20 --map-depth=2 --glr=0.001 --dlr=0.001 --cbase=16384 --kimg=30000

