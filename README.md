# TOM-Net
**[TOM-Net: Learning Transparent Object Matting from a Single Image, CVPR 2018 (Spotlight)](http://gychen.org/TOM-Net/)**,
<br>
[Guanying Chen](http://www.gychen.org)\*, [Kai Han](http://www.hankai.org/)\*, [Kwan-Yee K. Wong](http://i.cs.hku.hk/~kykwong/)
<br>
(\* equal contribution)

This paper addresses the problem of transparent object matting from a single image:
<br>
<p align="center">
    <img src='images/cvpr2018_tom-net.jpg' width="600" >
</p>


### Dependencies
TOM-Net is implemented in [Torch](http://torch.ch/) and tested with Ubuntu 14.04, please install Torch first following the [official document](http://torch.ch/docs/getting-started.html#_). 
- python 2.7 
- numpy
- cv2 
- CUDA-8.0  
- CUDNN v5.1
- Torch STN ([qassemoquab/stnbhwd](https://github.com/qassemoquab/stnbhwd))
    ```shell
    # Basic installation steps for stn
    git clone https://github.com/qassemoquab/stnbhwd.git
    cd stnbhwd
    luarocks make
    ```

## Overview:
We provide:

- Pretrained model
- Datasets: Train (40GB), Validation (196MB), Test (179MB)
- Code to test model on new images
- Evaluation code on both the validation and testing data
- Instructions to train the model 
- Example code for synthetic data rendering

## Testing
#### Download Pretrained Model
```
sh scritps/download_pretrained_model.sh
```

#### Test on New Images
```shell
# Replace ${gpu} with the selected GPU ID (starting from 0)

# Test a single image without having the background image
CUDA_VISIBLE_DEVICES=${gpu} th eval/run_model.lua -input_img images/bull.jpg 

# You can find the results in data/TOM-Net_model/
```

#### Evaluation on Synthetic Validation Data
```shell
# Download synthetic validation dataset
sh scripts/download_validation_dataset.sh

# Quantitatively evaluate TOM-Net on different categories of synthetic object 
# Replace ${class} with one of the four object categories (glass, water, lens, cplx)
CUDA_VISIBLE_DEVICES=${gpu} th eval/run_synth_data.lua -img_list ${class}.txt

# Similarly, you can find the results in data/TOM-Net_model/
```

#### Evaluation on Real Testing Data
```shell
# Download real testing dataset, 
sh scripts/download_testing_dataset.sh

# Test on sample images used in the paper
CUDA_VISIBLE_DEVICES=${gpu} th eval/run_model.lua -img_list Sample_paper.txt

# Quantitatively evaluate TOM-Net on different categories of real-world object 
# Replace ${class} with one of the four object categories (Glass, Water, Lens, Cplx)
CUDA_VISIBLE_DEVICES=${gpu} th eval/run_model.lua -img_list ${class}.txt  
```

## Training
To train a new TOM-Net model, you have to follow the following steps:
- Download the training data
```shell
# The size of the zipped training dataset is 40 GB and you need about 207 GB to unzip it.
sh scripts/download_training_dataset.sh
```

- Call `main.lua` to train CoarseNet on simple objects
```shell
CUDA_VISIBLE_DEVICES=$gpu th main.lua -train_list train_simple_98k.txt -nEpochs 13 -prefix 'simple'
# Please refer to opt.lua for more information about the training options

# You can find log file, checkpoints and visualization results in data/training/simple_*
```

- Call `main.lua` to train CoarseNet on both simple and complex objects
```shell
# Finetune CoarseNet with all of the data
CUDA_VISIBLE_DEVICES=$gpu th main.lua -train_list train_all_178k.txt -nEpochs 7 -prefix 'all' -retrain data/training/simple_*/checkpointdir/checkpoint13.t7

# You can find log file, checkpoints and visualization results in data/training/all_*
```

- Call `main_refine.lua` to train RefineNet on both simple and complex objects
```shell
CUDA_VISIBLE_DEVICES=$gpu th refine/main_refine.lua -train_list train_all_178k.txt -nEpochs 20 -coarse_net data/training/all_*/checkpointdir/checkpoint7.t7 
# Train RefineNet with all of the data
# Please refer to refine/opt_refine.lua for more information about the training options

# You can find log file, checkpoints and visualization results in data/training/all_*/refinement/
```

## Synthetic Data Rendering
Please refer to [TOM-Net_Rendering](https://github.com/guanyingc/TOM-Net_Rendering) for sample rendering codes.

## IJCV Extension

### TOM-Net + Trimap
```
# Testing
CUDA_VISIBLE_DEVICES=3 th eval/run_model.lua -input_root images/Trimap_demo -img_list img_trimap_list.txt -c_net data/Release_Trimap_model_Feb23_2019/CoarseNet_Trimap_19.t7 -r_net data/Release_Trimap_model_Feb23_2019/RefineNet_Trimap_10.t7 -in_trimap 

# Training CoarseNet
CUDA_VISIBLE_DEVICES=3 th main.lua -data_dir ~/TOM-Net_Synth_Train_178k/ -train_list train_all_178k.txt -in_trimap -nEpochs 20
# Training RefineNet
CUDA_VISIBLE_DEVICES=2 th refine/main_refine.lua -coarse_net data/training/Feb-23__CoarseNet_scale_h-512_crop_h-448_flow_w-0.010_mask_w-0.100_rho_w-1.000_img_w-1.000_lr-0.000100_Trimap_/checkpointdir/checkpoint20.t7 -data_dir ~/TOM-Net_Synth_Train_178k/ -train_list train_all_178k.txt -in_trimap -nEpochs 10
```

### TOM-Net + Background
```
# Testing on real data
CUDA_VISIBLE_DEVICES=3 th eval/run_model.lua -input_root ../TOM-Net_Train/data/datasets/TOM-Net_Real_Test_876 -c_net data/Release_Stereo_model_Jan23_2019/CoarseNet_with_bg_19.t7 -r_net data/Release_Stereo_model_Jan23_2019/RefineNet_with_bg_11.t7 -in_bg 
# Testing on synthetic data
CUDA_VISIBLE_DEVICES=3 th eval/run_synth_data.lua -input_root ../TOM-Net_Train/data/datasets/TOM-Net_Synth_Val_900/ -c_net data/Release_Stereo_model_Jan23_2019/CoarseNet_with_bg_19.t7 -r_net data/Release_Stereo_model_Jan23_2019/RefineNet_with_bg_11.t7 -in_bg

# Training CoarseNet
CUDA_VISIBLE_DEVICES=3 th main.lua -data_dir ~/TOM-Net_Synth_Train_178k/ -train_list train_all_178k.txt -in_bg -nEpochs 20
# Training RefineNet
CUDA_VISIBLE_DEVICES=2 th refine/main_refine.lua -coarse_net data/training/Feb-23__CoarseNet_scale_h-512_crop_h-448_flow_w-0.010_mask_w-0.100_rho_w-1.000_img_w-1.000_lr-0.000100_Trimap_/checkpointdir/checkpoint20.t7 -data_dir ~/TOM-Net_Synth_Train_178k/ -train_list train_all_178k.txt -in_bg -nEpochs 10
```
## Citation
If you find this code or the provided data useful in your research, please consider cite: 

```
@inproceedings{chen2018tomnet,
  title={TOM-Net: Learning Transparent Object Matting from a Single Image},
  author={Chen, Guanying and Han, Kai and Wong, Kwan-Yee K.},
  booktitle={CVPR},
  year={2018}
}
```


