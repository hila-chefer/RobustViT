# Official PyTorch implementation of **[Optimizing Relevance Maps of Vision Transformers Improves Robustness](https://arxiv.org/abs/2206.01161)**. 

This code allows to  finetune the explainability maps of Vision Transformers to enhance robustness.

## HuggingFace space + Colab notebook to run examples of the finetuned vs the original models:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hila-chefer/RobustViT/blob/master/RobustViT.ipynb)[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Hila/RobustViT)

## Updates:
06/05/2022 **Added a [HuggingFace Spaces demo](https://huggingface.co/spaces/Hila/RobustViT)**:
<p align="center">
  <img width="1200" height="750" src="hf_spaces.png">
</p>

## Method overview:
The method employs loss functions directly to the explainability maps to ensure that the model is focused mostly on the foreground of the image:
<p align="center">
  <img width="500" height="400" src="teaser.png">
</p>
Using a short finetuning process with only 3 labeled examples from 500 classes, our method imrpoves robustness of ViT models across different model sizes and training techniques, even when data augmentations/ regularization are applied.

## Model zoo
Below are links to download finetuned models for the base models of [ViT AugReg](https://arxiv.org/abs/2106.10270) (this is also the model that appears on [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)), vanilla ViT, and DeiT. 
These are also the weights used in our [colab notebook](https://colab.research.google.com/github/hila-chefer/RobustViT/blob/master/RobustViT.ipynb).
| Path | Description
| :--- | :----------
|[AugReg-B](https://drive.google.com/file/d/1jbWiuBrL4sKpAjG3x4oGbs3WOC2UdbIb/view?usp=sharing) | Finetuned ViT Augreg base model.
|[ViT-B](https://drive.google.com/file/d/1vDmuvbdLbYVAqWz6yVM4vT1Wdzt8KV-g/view?usp=sharing) | Finetuned vanilla ViT base model.
|[DeiT-B](https://drive.google.com/file/d/1DHKX_s8rVCDiX4pwnuCCZdGWsOl4SFMn/view?usp=sharing)| Finetuned DeiT base model.

## Producing Segmenataion Data
### Using ImageNet-S
To use the ImageNet-S labeled data, [download the `ImageNetS919` dataset](https://github.com/UnsupervisedSemanticSegmentation/ImageNet-S)

### Using TokenCut for unsupervised segmentation
1.  Clone the TokenCut project
    ```
    git clone https://github.com/YangtaoWANG95/TokenCut.git
    ```
2.  Install the dependencies
    Python 3.7, PyTorch 1.7.1 and CUDA 11.2. Please refer to the official installation. If CUDA 10.2 has been properly installed:
    ```
    pip install torch==1.7.1 torchvision==0.8.2
    ```
    Followed by
    ```
    pip install -r TokenCut/requirements.txt
    
3. Use the following command to extract the segmentation maps:
    ```
   python tokencut_generate_segmentation.py --img_path <PATH_TO_IMAGE> --out_dir <PATH_TO_OUTPUT_DIRECTORY>    
   ```


## Finetuning ViT models

To finetune a pretrained ViT model use the `imagenet_finetune.py` script. Notice to uncomment the import line containing the pretrained model you 
wish to finetune.

Usage example:

```bash
python imagenet_finetune.py --seg_data <PATH_TO_SEGMENTATION_DATA> --data <PATH_TO_IMAGENET> --gpu 0  --lr <LR> --lambda_seg <SEG> --lambda_acc <ACC> --lambda_background <BACK> --lambda_foreground <FORE>
```

Notes:

* For all models we use :
    * `lambda_seg=0.8`
    * `lambda_acc=0.2`
    * `lambda_background=2`
    * `lambda_foreground=0.3`
 * For **DeiT** models, a temprature is required as follows:
    * `temprature=0.65` for DeiT-B
    * `temprature=0.55` for DeiT-S
 * The learning rates per model are:
    * ViT-B: 3e-6
    * ViT-L: 9e-7
    * AR-S: 2e-6
    * AR-B: 6e-7
    * AR-L: 9e-7
    * DeiT-S: 1e-6
    * DeiT-B: 8e-7

## Baseline methods
Notice to uncomment the import line containing the pretrained model you wish to finetune in the code.

### GradMask
Run the following command: 
```bash
python imagenet_finetune_gradmask.py --seg_data <PATH_TO_SEGMENTATION_DATA> --data <PATH_TO_IMAGENET> --gpu 0  --lr <LR> --lambda_seg <SEG> --lambda_acc <ACC>
```
All hyperparameters for the different models can be found in section D of the supplementary material.

### Right for the Right Reasons
Run the following command: 
```bash
python imagenet_finetune_rrr.py --seg_data <PATH_TO_SEGMENTATION_DATA> --data <PATH_TO_IMAGENET> --gpu 0  --lr <LR> --lambda_seg <SEG> --lambda_acc <ACC>
```
All hyperparameters for the different models can be found in section D of the supplementary material.

## Evaluation

### Robustness Evaluation

1. Download the evaluation datasets: 
    * [INet-A](https://github.com/hendrycks/natural-adv-examples)
    * [INet-R](https://github.com/hendrycks/imagenet-r)
    * [INet-v2](https://github.com/modestyachts/ImageNetV2)
    * [ObjectNet](https://objectnet.dev/)
    * [SI-Score](https://github.com/google-research/si-score)

2. Run the following script to evaluate:
 
```bash
python imagenet_eval_robustness.py --data <PATH_TO_ROBUSTNESS_DATASET> --batch-size <BATCH_SIZE> --evaluate --checkpoint <PATH_TO_FINETUNED_CHECKPOINT>
```
* Notice to uncomment the import line containing the pretrained model you wish to evaluate in the code.
* To evaluate the original model simply omit the `checkpoint` parameter.
* For the INet-v2 dataset add `--isV2`.
* For the ObjectNet dataset add `--isObjectNet`.
* For the SI datasets add `--isSI`.

### Segmentation Evaluation
Our segmentation tests are based on the test in the official implementation of [Transformer Interpretability Beyond Attention Visualization](https://github.com/hila-chefer/Transformer-Explainability).
1. [Download the ImageNet segmentation test set](https://github.com/hila-chefer/Transformer-Explainability#section-a-segmentation-results).
2. Run the following script to evaluate:
 
 ```bash
PYTHONPATH=./:$PYTHONPATH python SegmentationTest/imagenet_seg_eval.py  --imagenet-seg-path <PATH_TO_gtsegs_ijcv.mat>
```
* Notice to uncomment the import line containing the pretrained model you wish to evaluate in the code.

### Credits
* The TokenCut code is built on top of [LOST](https://github.com/valeoai/LOST), [DINO](https://github.com/facebookresearch/dino), [Segswap](https://github.com/XiSHEN0220/SegSwap), and [Bilateral_Sovlver](https://github.com/poolio/bilateral_solver). 
* Our ViT code is based on the [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) repository.
* Our ImageNet finetuning code is based on [code from the official PyTorch repo](https://github.com/pytorch/examples/blob/main/imagenet/main.py).
* The code to convert ObjectNet classes to ImageNet classes was taken from [the torchprune repo](https://github.com/lucaslie/torchprune/blob/b753745b773c3ed259bf819d193ce8573d89efbb/src/torchprune/torchprune/util/datasets/objectnet.py).
* The code to convert SI-Score classes to ImageNet classes was taken from [the official implementation](https://github.com/google-research/si-score).

We would like to sincerely thank the authors for their great works. 

## Citing our paper
If you make use of our work, please cite our paper:
```
@InProceedings{chefer2022robustvit,
    author    = {Chefer, Hila and Schwartz, Idan and Wolf, Lior},
    title     = {Optimizing Relevance Maps of Vision Transformers Improves Robustness},
   journal={arXiv preprint arXiv: 2206.01161},
   year={2022}
}
```
