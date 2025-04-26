# FrontView-VTON

Diffusion-based Conditional Inpainting for High-Quality Virtual Try-On of Diffusion model with Appearance Flow. Based on [DCI-VTON-Virtual-Try-On](https://github.com/bcmi/DCI-VTON-Virtual-Try-On). Step-1 of MV-VTON.

### Requirements
Requires to install: **lora_diffusion** from https://github.com/cloneofsimo/lora

```bash
pip install git+https://github.com/diixo/lora.git
pip install diffusers==0.20.0
```

### 3D-rendering

* 3D-rendering with **vgg19_conv.pth**: https://github.com/halfjoe/3D-Portrait-Stylization + **vgg_caffe.py** loader.

Based on:
* https://github.com/daniilidis-group/neural_renderer


### Dress-code dataset

* Github official: https://github.com/aimagelab/dress-code
* [Request form](https://forms.gle/72Bpeh48P7zQimin7) with [description](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=47)


## Knowledgments

* [OpenPose](https://github.com/Hzzone/pytorch-openpose)
* [multimodal-garment-designer](https://github.com/aimagelab/multimodal-garment-designer)
* [MF-VITON: High-Fidelity Mask-Free Virtual Try-On with Minimal Input](https://arxiv.org/abs/2503.08650)
* [Improving Diffusion Models for Authentic Virtual Try-on in the Wild](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11626.pdf)
