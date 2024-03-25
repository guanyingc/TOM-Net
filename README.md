# TOM-Net
**[TOM-Net: Learning Transparent Object Matting from a Single Image, CVPR 2018 (Spotlight)](https://guanyingc.github.io/TOM-Net/)**,
<br>
[Guanying Chen](https://guanyingc.github.io)\*, [Kai Han](http://www.hankai.org/)\*, [Kwan-Yee K. Wong](http://i.cs.hku.hk/~kykwong/)
<br>
(\* equal contribution)

This paper addresses the problem of transparent object matting from a single image.
<br>
<p align="center">
    <img src='images/cvpr2018_tom-net.jpg' width="600" >
</p>


## Dependencies
TOM-Net is implemented in [Torch](http://torch.ch/) and tested with Ubuntu 14.04. Please install Torch first following the [official document](http://torch.ch/docs/getting-started.html#_). 
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

## Overview
We provide:

- Pretrained model
- Datasets: Train (40GB), Validation (196MB), Test (179MB)
- Code to test model on new images
- Evaluation code on both the validation and testing data
- Instructions to train the model 
- Example code for synthetic data rendering
- Code and models used in the journal extension <b>(New!)</b>

If the automatic downloading scripts are not working, please download the trained models and the introduced dataset from BaiduYun ([Models and Datasets](https://pan.baidu.com/s/1EHiiwK1BBlrXWGkqyYweaw?pwd=837t)).

## Testing
#### Download Pretrained Model
```
sh scripts/download_pretrained_model.sh
```
If the above command is not working, please manually download the trained models from BaiduYun (Models, Datasets) and put them in `./data/models/`.

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
To train a new TOM-Net model, please follow the following steps:
- Download the training data
```shell
# The size of the zipped training dataset is 40 GB and you need about 207 GB to unzip it.
sh scripts/download_training_dataset.sh
```

- Train CoarseNet on simple objects
```shell
CUDA_VISIBLE_DEVICES=$gpu th main.lua -train_list train_simple_98k.txt -nEpochs 13 -prefix 'simple'
# Please refer to opt.lua for more information about the training options

# You can find log file, checkpoints and visualization results in data/training/simple_*
```

- Train CoarseNet on both simple and complex objects
```shell
# Finetune CoarseNet with all of the data
CUDA_VISIBLE_DEVICES=$gpu th main.lua -train_list train_all_178k.txt -nEpochs 7 -prefix 'all' -retrain data/training/simple_*/checkpointdir/checkpoint13.t7

# You can find log file, checkpoints and visualization results in data/training/all_*
```

- Train RefineNet on both simple and complex objects
```shell
CUDA_VISIBLE_DEVICES=$gpu th refine/main_refine.lua -train_list train_all_178k.txt -nEpochs 20 -coarse_net data/training/all_*/checkpointdir/checkpoint7.t7 
# Train RefineNet with all of the data
# Please refer to refine/opt_refine.lua for more information about the training options

# You can find log file, checkpoints and visualization results in data/training/all_*/refinement/
```

## Synthetic Data Rendering
Please refer to [TOM-Net_Rendering](https://github.com/guanyingc/TOM-Net_Rendering) for sample rendering codes.

## Codes and Models Used in the Journal Extension (IJCV)
#### Test TOM-Net<sup>+Bg</sup> and TOM-Net<sup>+Trimap</sup> on Sample Images
```shell
# Download pretrained models
sh scripts/download_pretrained_models_IJCV.sh

# Test TOM-Net+Bg on sample images
CUDA_VISIBLE_DEVICES=${gpu} th eval/run_model.lua -input_root images/TOM-Net_with_Trimap_Bg_Samples/ -img_list img_bg_trimap_list.txt -in_bg -c_net data/TOM-Net_plus_Bg_Model/CoarseNet_plus_Bg.t7 -r_net data/TOM-Net_plus_Bg_Model/RefineNet_plus_Bg.t7 
# You can find the results in data/TOM-Net_plus_Bg_Model/*

# Test TOM-Net+Trimap on sample images
CUDA_VISIBLE_DEVICES=${gpu} th eval/run_model.lua -input_root images/TOM-Net_with_Trimap_Bg_Samples/ -img_list img_bg_trimap_list.txt -in_trimap -c_net data/TOM-Net_plus_Trimap_Model/CoarseNet_plus_Trimap.t7 -r_net data/TOM-Net_plus_Trimap_Model/RefineNet_plus_Trimap.t7 
# You can find the results in data/TOM-Net_plus_Trimap_Model/*
```

#### Train TOM-Net<sup>+Bg</sup> and TOM-Net<sup>+Trimap</sup> 
To train a new TOM-Net<sup>+Bg</sup> or TOM-Net<sup>+Trimap</sup> model, please follow the same procedures as training TOM-Net, except that you need to append `-in_bg` or `-in_trimap` at the end of the commands.

## Citation
If you find this code or the provided data useful in your research, please consider cite the following relevant paper(s): 

```
@inproceedings{chen2018tomnet,
  title={TOM-Net: Learning Transparent Object Matting from a Single Image},
  author={Chen, Guanying and Han, Kai and Wong, Kwan-Yee K.},
  booktitle={CVPR},
  year={2018}
}

@inproceedings{chen2019LTOM,
  title={Learning Transparent Object Matting},
  author={Chen, Guanying and Han, Kai and Wong, Kwan-Yee K.},
  booktitle={IJCV},
  year={2019}
}
```


