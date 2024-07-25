
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

## Installation
```
git clone https://github.com/ECoLab-POSTECH/TDQ_NeurIPS2023
cd TDQ_NeurIPS2023
conda env create -f environment.yaml
conda activate tdq
```

### Training TDQ
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
```

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

```
./scripts/download_first_stages.sh
./scripts/download_models.sh
```
- `./scripts/download_first_stages.sh` script is responsible for downloading and extracting first-stage models used in Latent Diffusion project.
  - It downloads a set of model files from the `ommer-lab.com` website and extracts them to the `./models/first_stage_models` directory.
  - The models downloaded include different variations of KL-diffusion (`kl-f4`, `kl-f8`, `kl-f16`, `kl-f32`), and Vector Quantized (`vq-f4`, `vq-f4-noattn`, `vq-f8`, `vq-f8-n256`, `vq-f16`) models.
- `download_models.sh` script is for downloading and extracting various pre-trained models used in the Latent Diffusion (LDM) framework.
  - The models are downloaded from the `ommer-lab.com` website and saved in the `./models/ldm` directory.
  - The script downloads models for different tasks, including CelebA-256, FFHQ-256, LSUN Churches-256, LSUN Bedrooms-256, Text-to-Image-256, CIN-256, Semantic Synthesis-512, Semantic Synthesis-256, Super-Resolution BSR, Layout-to-Image-OpenImages-256, and Inpainting-Big. 

### Evaluation 
```
./scripts/evaluate_fid.py
```
- `./scripts/evaluate_fid.py` evaluates the FID metric between two datasets.
  - This script supports evaluating FID for the CIFAR-10 train and LSUN churches-256 train dataset.
    ```
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


