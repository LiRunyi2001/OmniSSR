<div align="center">
<h3>[ECCV2024] OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model</h3>

[Runyi Li](https://lirunyi2001.github.io), [Xuhan Sheng](https://github.com/llstela/), [Weiqi Li](https://github.com/lwq20020127), [Jian Zhang](https://jianzhang.tech/)

School of Electronic and Computer Engineering, Peking University

[![arXiv](https://img.shields.io/badge/arXiv-2404.10312-b31b1b.svg)](https://arxiv.org/abs/2404.10312)
[![Home Page](https://img.shields.io/badge/Project-<Website>-blue.svg)](https://lirunyi2001.github.io/projects/omnissr)

This repository is the official implementation of OmniSSR, a *zero-shot* omni-directional image super-resolution framework based on Stable Diffusion. We propose Gradient Decomposition to guide the prior sampling of SD with the degradation operation.

<img src="__assets__\method_v2_00.jpg"/>

<img src="__assets__\Proj-Trans_00.jpg"/>
</div>

## Gallery

We have showcased some SR results generated by OmniSSR below. 

More results can be found on our [Project Page](https://lirunyi2001.github.io/projects/omnissr).

<img src="__assets__\teaser_00.jpg"/>

|               Ground-Truth (X2)               | DDRM | StableSR  |                OmniSSR (ours)                 |
|:---------------------------------------------:|:----------------------------------------------------:|:-------------------------------------------------:|:-------------------------------------------------:|
| <img src="__assets__\comparison\odisr-x2\gt_0067_5_00.jpg" width="300"/> |   <img src="__assets__\comparison\odisr-x2\ddrm_0067_out_5_00.jpg" width="300"/>    | <img width="300" src="__assets__\comparison\odisr-x2\stablesr_0067_5_00.jpg"> | <img width="300" src="__assets__\comparison\odisr-x2\ours_0067_5_00.jpg"> |


## To Do List
- [x] Release code
- [x] Release paper

##  Guidance for Inference

### Prepare Environment
You have 2 choices to prepare the environment.
1. You can follow [StableSR](https://github.com/IceClear/StableSR/) to prepare the environment, and pip install other packages after running the code.
2. or You can follow the guidance below:
```
# Step 1: clone the code
git clone https://github.com/LiRunyi2001/OmniSSR.git
cd OmniSSR

# Step 2: prepare conda env
conda create -n omnissr python=3.10
conda activate omnissr

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117  # make sure CUDA is compatible

pip install -r requirements.txt

# prepare CLIP and taming-transformers

mkdir src
cd src
git clone https://github.com/CompVis/taming-transformers.git
git clone https://github.com/openai/CLIP.git
cd taming-transformers
python setup.py develop  # install taming-transformers
cd ..
cd CLIP
python setup.py develop  # install CLIP
cd ../..  # now at OmniSSR dir
```

### Prepare Model Weights
Our Experiments are based on StableSR, using pretrained model weights **stablesr_000117.ckpt** and **vqgan_cfw_00011.ckpt**. You can download them at
https://huggingface.co/Iceclear/StableSR/tree/main.

### Prepare Dataset
We follow [OSRT](https://github.com/Fanghua-Yu/OSRT) to clean the ODI-SR and SUN360 datasets. 

You can refer to **make_clean_lau_dataset.py** for detailed implementation.

Datasets are available at [Baidu Netdisk](https://pan.baidu.com/s/1EruFzvF0G4YQ6gPsDKoCZA?pwd=lv1f) (valid code: **lv1f**)

The file tree of the dataset is as follows:

```
|-- lau_dataset_resize_clean
|   |-- odisr
|   |   |-- testing
|   |   |   |-- HR
|   |   |   |   |-- 0000.png
|   |   |   |   |-- 0001.png
|   |   |   |-- LR_erp
|   |   |   |   |-- X2.00
|   |   |   |   |-- X4.00
|   |   |   |   |   |-- 0000.png
|   |   |   |   |   |-- 0001.png
|   |-- sun_test
|   |   |-- HR
|   |   |   |-- 0000.png
|   |   |   |-- 0001.png
|   |   |-- LR_erp
|   |   |   |-- X2.00
|   |   |   |-- X4.00
|   |   |   |   |-- 0000.png
|   |   |   |   |-- 0001.png
```

### Run Inference

We list some more useful configurations for easy usage:

|    Argument     |                     Description                     |
|:---------------:|:---------------------------------------------------:|
|   `init-img`  |           path of input LR images                   |
|   `outdir`    |           path of output HR images                  |
|   `upscale`   |           upsamling scale for LR images             |
|  `dist_k_fold`|           (optional) for distributed inference. <br> e.g.: **3/5** means deviding whole images into **5** folds, and only inference the **3rd** fold.             |

A complete inference command line is as follows:
```
CUDA_VISIBLE_DEVICES=2  \
python scripts/test_sr_val_ddpm_text_T_oldcanvas_TAN-OmniSSR-TAN.py \
--config configs/stableSRNew/v2-finetune_text_T_512_pano.yaml \
--ckpt {path-to-stablesr}/stablesr_000117.ckpt \
--vqgan_ckpt {path-to-stablesr}/vqgan_cfw_00011.ckpt \
--init-img {path-to-stablesr}/lau_dataset_resize_clean/sun_test/LR_erp/X2.00 \
--outdir results/sun-test-x2/test_OmniSSR \ 
--ddpm_steps 200  # defaut setting of StableSR \
--dec_w 0.5  # defaut setting of StableSR \
--colorfix_type adain  # defaut setting of StableSR \
--upscale 2.0 \
--input_size 512  # defaut setting of StableSR \
--dist_k_fold 1/2
```

The output images are in {outdir}/erp_output.
You should run **post_erp_rnd.ipynb** to get the final results.

## Contact Us
**Runyi Li**: [lirunyi@stu.pku.edu.cn](mailto:lirunyi@stu.pku.edu.cn)
**Xuhan Sheng**: [shengxuhan@stu.pku.edu.cn](mailto:shengxuhan@stu.pku.edu.cn)


## Acknowledgements
Codebase built upon [StableSR](https://github.com/IceClear/StableSR/) and [DDNM](https://github.com/wyhuai/DDNM).

## BibTeX
```
@article{li2024omnissr,
  title={OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model},
  author={Li, Runyi and Sheng, Xuhan and Li, Weiqi and Zhang, Jian},
  journal={arXiv preprint arXiv:2404.10312},
  year={2024}
}
```