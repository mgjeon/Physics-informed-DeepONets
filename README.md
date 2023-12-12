# Physics-informed-DeepONet

```
git clone https://github.com/mgjeon/Physics-informed-DeepONets
cd Physics-informed-DeepONets
git remote add upstream https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets
git checkout learn
git fetch upstream
git merge upstream/main
git push origin learn
```

```
mamba create -n pidon
mamba activate pidon
mamba update -y mamba
mamba install -y python=3.10 ipykernel
pip install --upgrade pip
mamba install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
cd ~/miniforge3/envs/pidon/lib
ln -sfn libnvrtc.so.11.8.89 libnvrtc.so
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install -e .
```