# Latent Diffusion Models
[CompVis Source Code](https://github.com/CompVis/latent-diffusion)

April 2022
- 1.45B latent diffusion LAION model was integrated into Huuggingface Spaces using Gradio.
  - You can try out the web demo.
  - 1.45B model trained on the LAION-400M database.

## How to Create Images beyond $256 \times 256$
For certain inputs, simply ruunning the model in a convolutional fashion on larger features than it was trained on can sometimes result in interesting results.
To try it out, tune the `H` and `W` arguments (which will be integer-divided by `8` in order to calculate the corresponding latent size), run the following command:
```
python scripts/txt2img.py --prompt "a sunset behind a mountain range, vector image" --ddim_eta 1.0 --n_samples 1 --n_iter 1 --H 384 --W 1024 --scale 5.0
```
This creates a sample of size $384 \times 1024$. Note, however, that controllability is reduced compared to the $256 \times 256$ setting.
![image](https://github.com/user-attachments/assets/99d64807-9c8e-4350-bede-7c8325ece012)


## Train Your Own LDMs


## Git Commit
```bash
git status
git add .
git commit -m "updating README.md"
# list all the branches in your repository, the current branch is marked with asterisk
git branch
git branch --show-current


# push your changes to GitHub
git push origin main
git push origin master
```


## Model Training 

Logs and checkpoints for trained models are saved to `logs/<START_DATE_AND_TIME>_<config_spec>`.


### Training Autoencoder Models

Configs for training KL-regularized autoencoder on ImageNet are provided at `configs/autoencoder`. Training can be started by running

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --base configs/autoencoder/autoencoder_kl_8x8x64.yaml -t --gpus 0,
```

- `autoencoder_kl_8x8x64.yaml` is config for $f=32, d=64$. 
- `autoencoder_kl_16x16x16.yaml` is config for $f=16,d=16$.
- `autoencoder_kl_32x32x4.yaml` is config for $f=8,d=4$.
- `autoencoder_kl_64x64x3.yaml` is config for $f=4,d=3$.

For VQ-regularized models, see the taming-transformers repository.

### Training LDMs

In `configs/latent-diffusion` we provide configs for training LDMs on the LSUN-churches, CelebA-HQ, FFHQ and ImageNet datasets. Training can be started by running

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --base configs/latent-diffusion/lsun_bedrooms-ldm-vq-4.yaml -t --gpus 0,
```

- `celebahq-ldm-vq-4.yaml` is $f=4$, VQ-reg. autoencoder, spatial size $64 \times 64 \times 3$.
- `ffhq-ldm-vq-4.yaml` is $f=4$, VQ-reg. autoencoder, spatial size $64 \times 64 \times 3$.




















