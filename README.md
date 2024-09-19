# GTA-Crime
This repo contains the Pytorch implementation of our paper:
> [**GTA-Crime: Leveraging Synthetic Video and Feature-Level Domain Adaptation for Enhanced Fatal Violence Detection**](paper link)
>
> Seongho Kim, Sejong Ryu, Hyoukjun You, Je Hyeong Hong

- **Submitted at ICASSP 2025.**

![overall pipeline](overall_pipeline.png)

## Enviroment
- Python 3.12.4
- PyTorch 2.4.0
- Torchvision 0.19.0
- Pytorch-cuda 12.4

## Dataset

**Please download the GTA-Crime dataset from links below:**

> [**GTA-Crime videos on Google Drive**](https://drive.google.com/file/d/14mA5jgSIlfGdE6P-PgOr0bFCOyDSxGhc/view?usp=sharing)

**Extracted I3d and CLIP features for GTA-Crime dataset**
> [**GTA-Crime i3d features on Google Drive**](https://drive.google.com/file/d/14CQoSPS0iRwkTfA8AHcLwImruuUPAUdu/view?usp=sharing)
> 
> [**GTA-Crime CLIP features on Google Drive**](https://drive.google.com/file/d/1fK5B5tJ-dVDSsS8LLawF2hecnMXfBouI/view?usp=sharing)

**We use the extracted I3D and CLIP features for UCF-Crime datasets from the following works:**
> [**UCF-Crime 10-crop I3D features**](https://github.com/Roc-Ng/DeepMIL)
>
> [**UCF-Crime 10-crop CLIP features**](https://github.com/nwpu-zxr/VadCLIP)

- Change the file paths to the download datasets above in ```list/ucf_CLIP_rgb.csv``` and ```list/newGTA.csv```.

## Synthetic Dataset Construction
We extend the code from [GTA5Event](https://github.com/RicoMontulet/GTA5Event) to create fatal scenarios involving stabbing and shooting.

Instructions for creating the dataset are in [CONSTRUCTION.md](https://github.com/ta-ho/GTA-Crime/blob/main/CONSTRUCTION.md#gta-crime-construction).

## Stage1. Feature-level domain adaptation
The code for stage 1 of our method can be found in the directory ```stage1```. To train the model with the default arguments, run the code below
- using WGAN-GP loss
    ```
    cd stage1/wgan_gp
    python main.py
    ```
- using CycleGAN loss
    ```
    cd stage1/cycgan
    python ucf_each.py
    ```

## Stage2. VAD model training and testing
The code for each model in stage2 is in derectory ```stage2```. To train on VAD model, place the model trained on WGAN-GP/CycleGAN (from stage 1) to ```stage2/VadCLIP/weights```. Then, run the code below
- VadCLIP
    ```
    cd stage2/VadCLIP
    python ./src/ucf_train.py
    ```
- CLIP-TSA
    ```
    cd stage2/CLIP-TSA
    python main.py
    ```
- UR-DMU
    ```
    cd stage2/UR-DMU
    python ucf_main.py
    ```
- MGFN
    ```
    cd stage2/MGFN
    python main.py
    ```
- RTFM
    ```
    cd stage2/RTFM
    python main.py
    ```

## References
We referenced the repos below for the code.
* [VadCLIP](https://github.com/nwpu-zxr/VadCLIP)
* [CLIP-TSA](https://github.com/joos2010kj/CLIP-TSA/tree/main)
* [UR-DMU](https://github.com/henrryzh1/UR-DMU)
* [MGFN](https://github.com/carolchenyx/MGFN.)
* [RTFM](https://github.com/tianyu0207/RTFM/tree/main)

## Result on VAD models (AUC)
| Method  |    UCF    | UCF+GTA(w/o DA)|UCF+GTA(w/ WGAN-GP)|UCF+GTA(w/ CycleGAN)|
| ------  | :-------: | :---------: | :-------: | :-------: |
| RTFM    |  85.43    |  84.98  |  87.27 | 87.28|
| UR-DMU  | 86.04 | 81.39  | 86.47 | 86.35 |
| MGFN    | 82.55 | 79.37  | 83.64 | 84.35 |
| CLIP-TSA| 78.75 | 81.62  | 82.66 | 81.65 |
| VadCLIP| 73.59 | 74.60   | 74.79 | 76.84 |
