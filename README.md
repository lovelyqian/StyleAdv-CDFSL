# 1 StyleAdv-CDFSL

Repository for the CVPR-2023 paper : StyleAdv: Meta Style Adversarial Training for Cross-Domain Few-Shot Learning

[[Paper](https://arxiv.org/pdf/2302.09309)], [[Presentation Video on Bilibili](https://www.bilibili.com/video/BV1th4y1s78H/?spm_id_from=333.999.0.0&vd_source=668a0bb77d7d7b855bde68ecea1232e7)], [[Presentation Video on Youtube](https://youtu.be/YB-S2YF22mc)]

<img width="470" alt="image" src="https://github.com/lovelyqian/StyleAdv-CDFSL/assets/49612387/133c5248-1728-4f6e-a49c-6a7767f3a7ea">

# 2 Setup

## 2.1 conda env & code

```
# conda env
conda create --name py36 python=3.6
conda activate py36
conda install pytorch torchvision -c pytorch
conda install pandas
pip3 install scipy>=1.3.2
pip3 install tensorboardX>=1.4
pip3 install h5py>=2.9.0
pip3 install tensorboard
pip3 install timm
pip3 install opencv-python==4.5.5.62
pip3 install ml-collections
# code
git clone https://github.com/lovelyqian/StyleAdv-CDFSL
cd StyleAdv-CDFSL
```

## 2.2 Datasets

We use the mini-Imagenet as the single source dataset, and use cub, cars, places, plantae, ChestX, ISIC, EuroSAT, and CropDisease as novel target datasets.

For the mini-Imagenet, cub, cars, places, and plantae, we refer to the [FWT](https://github.com/hytseng0509/CrossDomainFewShot) repo.

For the ChestX, ISIC, EuroSAT, and CropDisease, we refer to the [BS-CDFSL](https://github.com/IBM/cdfsl-benchmark) repo.

If you can't find the Plantae dataset, we provide at [here](https://pan.baidu.com/s/1ZUVQfw-KEvJuTRiYco39-Q?pwd=0000), PIN: 0000, please cite its paper.

if you can't find the splits (base/val/novel.json), we provide them [here](https://drive.google.com/drive/folders/17xqSY3HDXB-3lSPYlklbCTok3H3nS0Pp?usp=sharing)

You should first set `cconfig_bscdfsl_dir.py` to your correct dir.

# 3 StyleAdv based on ResNet

## 3.1 meta-train StyleAdv

Our method aims at improving the generalization ability of models, we apply the style attack and adversarial training during the meta-train stage. Once the model is meta-trained, it can be used for inference on different novel target datasets directly.

Taking 5-way 1-shot as an example, the meta-train can be done as,

```
python3 metatrain_StyleAdv_RN.py --dataset miniImagenet --name exp-name --train_aug --warmup baseline --n_shot 1 --stop_epoch 200
```

- We integrate the testing into the training, and the testing results can be found on `output/checkpoints/exp-name/acc*.txt`;
- We set a probability `$p_{skip}$` for randomly skipping the attack, the value of it can be modified in `methods/tool_func.py`;
- We also provide our meta-trained ckps in `output/checkpoints/StyleAdv-RN-1shot` and `output/checkpoints/StyleAdv-RN-5shot`;

## 3.2 fine-tune the meta-trained StyleAdv

Though not necessary, for better performance, you may further fine-tune the meta-trained models on the target sets.

Taking 5-way 1-shot as an example, the fine-tuning on cars can be done as,

```
python3 finetune_StyleAdv_RN.py --testset cars --name exp-FT --train_aug --n_shot 1 --finetune_epoch 10 --resume_dir StyleAdv-RN-1shot --resume_epoch -1
```

- The finetuning is very sensitive to the `fintune_epoch` and `finetune_lr`;
- The value of `finetune_lr` can be modified in `finetune_StyleAdv_RN.py` :(sorry for not organizing the code very well;
- As attached in the [supplementary materials of paper](https://arxiv.org/pdf/2302.09309), we set the `finetune_epoch` and `finetune_lr` as:

  | Backbone 	| Task 	| Optimizer 	| finetune_epoch 	| finetune_lr 	|
  |:---:	|---	|---	|:---:	|---	|
  | RN10 	| 5-way 5-shot 	| Adam 	| 50 	| {0,0.001} 	|
  | RN10 	| 5-way 1-shot 	| Adam 	| 10 	| {0,0.005} 	|

# 4 StyleAdv based on ViT

You can use vit model in local path, for example vit-small like

`path/to/StyleAdv-CDFSL/checkpoints/pretrained_models/dino_deitsmall16_pretrain.pth`.

Or you can use the automatic way, the code will download the pretrained weight automaticly.

You can find it in 'path/to/StyleAdv-CDFSL/methods/load_ViT_models.py'

## 4.1 meta-train StyleAdv-ViT

To meta-train StyleAdv-ViT, correct dir to miniImagenet should be set in `path/to/StyleAdv-CDFSL/data/pmf_datasets/mini_imagenet.py`

The meta-train can be done as:

```
python metatrain_StyleAdv_ViT.py --output output/{output_dir} --dataset mini_imagenet --epoch 20 --lr 5e-5 --arch dino_small_patch16 --device cuda:0 --nSupport 1 --fp16
```

## 4.2 fine-tune the meta-trained StyleAdv-ViT

To fine-tune the meta-trained StyleAdv-ViT, `pmf_pretrained_ckp` should be set to correct directory of the meta-train StyleAdv-ViT checkpoint, which can be modified in `finetune_StyleAdv_ViT.py`.

The fine-tuning on EuroSAT can be done as:

```
python finetune_StyleAdv_ViT.py --testset EuroSAT --finetune_epoch 20 --model VitSmall --name finetune-with-ata-meta-tuning --n_shot 1
```

# 5 Citing

If you find our paper or this code useful for your research, please considering cite us (●°u°●)」:

```
@inproceedings{fu2023styleadv,
  title={StyleAdv: Meta Style Adversarial Training for Cross-Domain Few-Shot Learning},
  author={Fu, Yuqian and Xie, Yu and Fu, Yanwei and Jiang, Yu-Gang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={24575--24584},
  year={2023}
}
```

（if you are generally interested in CD-FSL task, we also have some other closely related works, attention and citation will be super appreciated!

```
@inproceedings{fu2025cross,
  title={Cross-domain few-shot object detection via enhanced open-set object detector},
  author={Fu, Yuqian and Wang, Yu and Pan, Yixuan and Huai, Lian and Qiu, Xingyu and Shangguan, Zeyu and Liu, Tong and Fu, Yanwei and Van Gool, Luc and Jiang, Xingqun},
  booktitle={European Conference on Computer Vision},
  pages={247--264},
  year={2025},
  organization={Springer}
}

@article{fu2022generalized,
  title={Generalized meta-fdmixup: Cross-domain few-shot learning guided by labeled target data},
  author={Fu, Yuqian and Fu, Yanwei and Chen, Jingjing and Jiang, Yu-Gang},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={7078--7090},
  year={2022},
  publisher={IEEE}
}

@inproceedings{fu2022me,
  title={Me-d2n: Multi-expert domain decompositional network for cross-domain few-shot learning},
  author={Fu, Yuqian and Xie, Yu and Fu, Yanwei and Chen, Jingjing and Jiang, Yu-Gang},
  booktitle={Proceedings of the 30th ACM international conference on multimedia},
  pages={6609--6617},
  year={2022}
}
```

# 6 Acknowledge

Our code is built upon the implementation of [FWT](https://github.com/hytseng0509/CrossDomainFewShot), [ATA](https://github.com/Haoqing-Wang/CDFSL-ATA), and [PMF](https://github.com/hushell/pmf_cvpr22). Thanks for their work.
