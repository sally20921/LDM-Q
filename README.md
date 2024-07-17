
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