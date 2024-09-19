
# Temporal Dynamic Quantization for Diffusion Models <br> [[Paper]](https://openreview.net/pdf?id=D1sECc9fiG) [[Slides]](https://neurips.cc/media/neurips-2023/Slides/72396_MFC2QO6.pdf) [[Poster]](https://neurips.cc/media/PosterPDFs/NeurIPS%202023/72396.png?t=1701674987.6762419)
#### [Junhyuk So*](https://github.com/junhyukso), [Jungwon Lee*](https://github.com/Jungwon-Lee), Daehyun Ahn, Hyungjun Kim and Eunhyeok Park

<br>

This is official PyTorch implementation of NeurIPS 2023 paper [Temporal Dynamic Quantization for Diffusion Models](https://openreview.net/pdf?id=D1sECc9fiG)

## Overview
__Temporal Dynamic Quantization (TDQ)__ is a novel quantization method for diffusion model that dynamically predicts quantization interval using time information, without any overhead. Using static quantizers for diffusion model quantization suffers from poor performance due to time-varying activation. Our method overcomes this problem by dynamically predicting the quantization parameters using temporal information. Unlike the common dynamic quantization methods, our method has no overhead by utilizing only temporal information.
<p align="center">
<img src=assets/overview.png />
</p>

## Getting Started

### Setting Up the Environment

#### Setting Up conda
1. Install pre-requisites
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y curl wget bzip2 libssl-dev zlib1g-dev \
    libreadline-dev libsqlite3-dev llvm libncursesw5-dev xz-utils tk-dev \
    libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
   ```
2. Download & Run miniconda installer
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   source ~/.bashrc
   ```

#### Tip
As always, when installing PyTorch, always do:
```
pip install light-the-torch
ltt install torch
```

```
git clone https://github.com/sally20921/TDQ
cd TDQ
conda env create -f environment.yaml
conda activate tdq
```

```
python -m venv tdq
source tdq/bin/activate
# Activate on Linux/macOS
# .\tdq\Scripts\activate
# Activate on Windows
pip install -r requirements.txt

```

### Downloading Pre-Trained Models
```bash
chmod +x scripts/download_first_stages.sh
chmod +x scripts/download_models.sh

./scripts/download_first_stages.sh
./scripts/download_models.sh
```

### Setting Directory Path

- Change the `./configs` directory to appropriate value.



### Generate Samples
- Use the `scripts/sample_diffusion.py` script to generate samples.
- This script uses the DDIM sampling method by default. 
```bash
python scripts/sample_diffusion.py --n_samples 10 --batch_size 2

# parameters
# --custom_steps : number of steps
# --eta : DDIM sampling eta value


```

### Generate Sample Images from Text Prompts

- Use the `scripts/txt2img.py` script to generate images from text prompts.
- This script supports various sampling methods, including DDIM, PLMS, and DPM Solver.
```bash
python scripts/txt2img.py --prompt "leopard hunting impala" --outdir outputs/txt2img-samples --ddim_steps 50
```





### Training TDQ

#### Preparing Dataset 

The dataset preparation is pretty much prepared the same way as in Stable Diffusion [https://github.com/CompVis/latent-diffusion].



- LSUN
The LSUN datasets can be conveniently downloaded via the script available here. We performed a custom split into training and validation images, and provide the corresponding filenames at https://ommer-lab.com/files/lsun.zip. After downloading, extract them to ./data/lsun. The beds/cats/churches subsets should also be placed/symlinked at ./data/lsun/bedrooms/./data/lsun/cats/./data/lsun/churches, respectively.

- LSUN Churches 256


FP checkpoint can be downloaded in [here](https://github.com/CompVis/latent-diffusion).
```
python main.py \
    --base configs/lsun_churches-ldm-kl-8.yaml \
    --ckpt models/ldm/lsun_churches256/model.ckpt \
    --train True \
    --scale_lr False \
    --n_bit_w ${w_bit} \
    --n_bit_a ${a_bit} \
    --name churches_W${w_bit}A${a_bit}_TDQ \
    --use_dynamic

## starting main.py with `--scale_lr False` set to target_lr by base_learning_rate = 5.0e-5
## in order for training to work, you need to change directory path in `configs/lsun_churches-ldm-kl-8.yaml`.

```
#### Running in a Multi-GPU system
- Use the `DDP` accelerator provided by PyTorch Lightning.
  ```
  pip install pytorch-lightning
  ```
```
python main.py --base configs/lsun_churches-ldm-kl-8.yaml --ckpt models/ldm/lsun_churches256/model.ckpt --train True --scale_lr False --n_bit_w 8 --n_bit_a 4 --name churches_W8A4_TDQ --use_dynamic --gpus 0,1,2,3,4,5,6,7 --accelerator ddp
```

#### Resuming Training from Checkpoint

```
python main.py --base configs/lsun_churches-ldm-kl-8.yaml --ckpt models/ldm/lsun_churches256/model.ckpt --train True --scale_lr False --n_bit_w 8 --n_bit_a 4 --gpu 0,1,2,3,4,5,6,7 --accelerator ddp --name churches_W8A4_TDQ --resume_from_checkpoint ./logs/2024-09-15T02-12-25_churches_W8A4_TDQ/checkpoints/last.ckpt
```

#### Saving Training and Error Message for Analysis

```bash
python main.py --base configs/lsun_churches-ldm-kl-8.yaml --ckpt models/ldm/lsun_churches256/model.ckpt --train True --scale_lr False --n_bit_w 4 --n_bit_a 8 --name churches_W4A8_TDQ --gpu 0,1,2,3,4,5,6,7 --accelerator ddp 2>&1 | tee error_logs/error_log_$(date +"%Y-%m-%d_%H-%M-%S").md
```

- `2>&1`: This redirects standard error (file descriptor 2) to standard output (file descriptor 1). This means that any error message will be sent to the same place as the regular output of your script.
- `| tee error_logs/error_log_$(date +"%Y-%m-%d_%H-%M-%S").md`: This pipes the combined output (both standard output and the redirected standard error) to the `tee` command. `tee` will then both display this output on the console and write it to the specified log file. 
- Make sure the `error_logs` directory exists. Otherwise, `tee` will fail to create the log file.


The base configuration in `lsun_churches-ldm-kl-8.yaml` is set as follows:

- base learning rate: 5.0e-5
- target: `ldm.models.diffusion.ddpm.LatentDiffusion`
  
- key parameter values:
  - linear start: 0.0015
  - linear end: 0.0155
  - timestep: 1000
  - loss_type: `l1`
  - image size: 32
  - channels: 4
  - scale by std: `False`
  - scale factor: 0.2458

- scheduler configuration:
  - target: `ldm.lr_scheduler.LambdaLinearScheduler`

- U-Net configuration:
  - `ldm.modules.diffusionmodules.openaimodel.UNetModel`
  - image size: 32
  - in channel: 4
  - out channel: 4
  - model channel: 192
  - attention resolution: [1,2,4,8]
  - number of ResNet block: 2
  - 

### Sampling from the model
```
python sample.py \
    --ckpt ${ckpt} \
    --config configs/lsun_churches-ldm-kl-8.yaml  \
    --logdir samples \
    --n_samples 10 \
    --custom_steps 100 \
    --gpus 0 \
    --batch_size 100 \
    --name test
```

## Citation
```
@inproceedings{
    so2023temporal,
    title={Temporal Dynamic Quantization for Diffusion Models},
    author={Junhyuk So and Jungwon Lee and Daehyun Ahn and Hyungjun Kim and Eunhyeok Park},
    booktitle={Advances in Neural Information Processing Systems},
    year={2023}
}
```

## Acknowledgments 
- Our codebase is based on [CompVis's Latent Diffusion codebase](https://github.com/CompVis/latent-diffusion).

## How to Use the Scripts


### Model Download and Extraction

```bash
chmod +x scripts/download_first_stages.sh
chmod +x scripts/download_models.sh

./scripts/download_first_stages.sh
./scripts/download_models.sh

## `download_models.sh` stores checkpoint in:
## `models/ldm/celeba256/celeba-256.zip`, 'models/ldm/ffhq256/ffhq-256.zip`, `models/ldm/lsun_churches256/lsun_churches-256.zip`.

## `download_first_stages.sh` stores checkpoint in:
## `models/first_stage_models/kl-f4/model.zip`, `models/first_stage_models/kl-f8/model.zip`, `models/first_stage_models/kl-f16/model.zip`
```
- `./scripts/download_first_stages.sh` script is responsible for downloading and extracting first-stage models used in Latent Diffusion project.
  - It downloads a set of model files from the `ommer-lab.com` website and extracts them to the `./models/first_stage_models` directory.
  - The models downloaded include different variations of KL-diffusion (`kl-f4`, `kl-f8`, `kl-f16`, `kl-f32`), and Vector Quantized (`vq-f4`, `vq-f4-noattn`, `vq-f8`, `vq-f8-n256`, `vq-f16`) models.
- `download_models.sh` script is for downloading and extracting various pre-trained models used in the Latent Diffusion (LDM) framework.
  - The models are downloaded from the `ommer-lab.com` website and saved in the `./models/ldm` directory.
  - The script downloads models for different tasks, including CelebA-256, FFHQ-256, LSUN Churches-256, LSUN Bedrooms-256, Text-to-Image-256, CIN-256, Semantic Synthesis-512, Semantic Synthesis-256, Super-Resolution BSR, Layout-to-Image-OpenImages-256, and Inpainting-Big. 

### Evaluation 

```bash
./scripts/evaluate_fid.py
```
Here's an example command to run the evaluation script:

```bash
python ./scripts/evaluate_fid.py \
    --logdir /SSD/stable_diffusion/QAT/samples \
    --dataset lsun_churches256 \
    --gpu 0 \
`   --name W4A8
```
- Ensure the model is quantized. Make sure you have a quantized model with W4A8 settings. This involves setting the number of bits for weights (`n_bit_w`) to `4` and activations (`n_bit_a`) to `8`.
- Download the required models. Ensure that you have downloaded the necessary models using the `download_model.sh` and `download_first_stages.sh` scripts.
- `--logdir`: directory where the samples are stored.
- `--dataset`: the dataset to evaluate against, in this case, `lsun_churches256`.
- `--gpu`: the gpu number to use.
- `--name`: the name of the model. I used `W4A8` to indicate the quantization settings.
- 

- `./scripts/evaluate_fid.py` evaluates the FID metric between two datasets.
  - This script supports evaluating FID for the CIFAR-10 train and LSUN churches-256 train dataset.
    ```python
    def get_dataset(opt):
        if opt.dataset == 'cifar-10' or opt.dataset == 'CIFAR-10':
            return 'cifar10-train'
        elif opt.dataset == 'lsun_churches256' or opt.dataset == 'LSUN_churches256' or opt.dataset == 'lsun_churches' or opt.dataset == 'LSUN-churches':
            opt.dataset = 'lsun_churches256'
            config_path = "models/ldm/lsun_churches256/config.yaml"
    ```
  - `validate()` function as the main entry point for evaluating FID metric.
    - creates an `EvalDataset` instance using the `opt.sample_path` argument
    - calls the `torch_fidelity.calculate_metrics()` function to compute the FID, Inception Score (IS) and other metrics between the two datasets.
  - **`torch_fidelity` library is used to provide a convenient interface for computing various image quality metrics.**
    - `torch_fidelity.calculate_metrics()`: `torch_fidelity.calculate_metrics(input1, input2, batch_size=256, cache_root, cache=True, datasets_root, cuda=True, isc=True, fid=True, kid=False, samples_shuffle=False, verbose=True,)`
- The computed metrics are printed to the console. 

###  Image Generation and Manipulation
The `./scripts/` directory contains several scripts that provide functionality for generating and manipulating images using pre-trained Stable Diffusion model. 
These scripts support various sampling methods, such as DDIM, PLMS, and DPM solver, and allow for image-to-image translation, inpainting, and text-to-image generation. 

```bash
python scripts/sample_diffusion.py --resume models/ldm/celeba256/model.ckpt --n_samples 50000 --eta 1.0 --vanilla_sample --logdir extra/logdir --custom_steps 50 --batch_size 10
```
- The `sample_diffusion.py` script generates samples using a pre-trained diffusion model, supporting both vanilla DDPM sampling and DDIM sampling.
- The `sample_diffusion.py` script also provides utility functions for rescaling, converting, and saving generated samples. 

```bash
./scripts/txt2img.py
```
- The `txt2img.py` script generates images from text prompts using a pre-trained Stable Diffusion model.
- `txt2img.py` supports various sampling methods, including DDIM, PLMS, and DPM Solver.
- `txt2img.py` also supports functionality for checking the generated images for NSFW content and adding watermarks.

```bash
./scripts/img2img.py
```
- `img2img.py` script generates variations of an input image using a pre-trained Stable Diffusion model.
- `img2img.py` script uses the `DDIMSampler` class from the `ldm/models/diffusion/ddim.py` module to perform DDIM sampling supporting both GPU and CPU acceleration.
- 

## Directory Structure

```
TDQ/
├── README.md
├── environment.yaml
├── main.py
├── setup.py
├── sample.py
├── ldm/
│   ├── data/
│   │   ├── base.py
│   │   ├── cifar10.py
│   │   ├── imagenet.py
│   │   ├── laion.py
│   │   ├── lsun.py
│   │   └── synthetic.py
│   ├── models/
|   │   ├── diffusion/
|   │   │   ├── __init__.py
|   │   │   ├── classifier.py
|   │   │   ├── ddim.py
|   │   │   ├── ddpm.py
|   │   │   ├── plms.py
|   |   |   |__ dpm_solver/
│   │   └── autoencoder.py
│   ├── modules/
│   │   ├── attention.py
|   |   ├── ema.py
|   |   ├── x_transformer.py
│   │   ├── diffusionmodules/
|   |   ├── distributions/
|   |   ├── encoders/
|   |   ├── image_degradation/
│   │   └── losses/
│   ├── lr_scheduler.py
│   └── util.py
├── scripts/
│   ├── script1.py
│   └── script2.py
├── configs/
│   ├── script1.py
│   └── script2.py
└── models/
    ├── test_main.py
    ├── test_module1.py
    └── test_module2.py
```

## Useful Resources
- pdf reading: Coral AI
- paper reading: typeset.io
- doc wiki: https://wiki.mutable.ai/sally20921/TDQ
- Stable Diffusion GUI txt2img

- [Rethinking How to Train Diffusion Models](https://developer.nvidia.com/blog/rethinking-how-to-train-diffusion-models/#:~:text=Exponential%20moving%20averages&The%20idea%20is%20to%20keep,%E2%80%9Crecent%E2%80%9D%20weights%20during%20training)
- [Training Models with Billions of Parameters](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/)
