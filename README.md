# CRA-CNN

Official MATLAB implementation of the Contrastively-reinforced Attention Convolutional Neural Network (CRA-CNN). CRA-CNN uses an additional network to reinforce the attention awareness of deep activations during the training procedure for the fine-grained image classification task. The additional network is removed at the inference phase to save computation costs. You may check more details in our [BMVC paper](https://www.bmvc2020-conference.com/assets/papers/0656.pdf) if you are interested in our work.

![The overview of CRA-CNN.](https://github.com/Dichao-Liu/CRA-CNN/blob/main/CRA-CNN.png)

## Requirements

The scripts need the following dependencies pre-installed:
 
[matconvnet-1.0-beta25](https://github.com/vlfeat/matconvnet)

[mcnExtraLayers](https://github.com/albanie/mcnExtraLayers)

[mcnPyTorch](https://github.com/albanie/mcnPyTorch)

[autonn](https://github.com/vlfeat/autonn)

Save the folders as:
```
CRA-CNN
├── CUB_Bird
├── layers
├── autonn-master
├── matconvnet-1.0-beta25
├── mcnExtraLayers-master
├── mcnPyTorch-master
```
## Usage
Unzip the files `bird-res50-biAtt.zip` and `bird-res50-biAtt-PRE.zip`, and you can obtain `bird-res50-biAtt.mat` and `bird-res50-biAtt-PRE.mat`. `bird-res50-biAtt.mat` is the CRA-CNN for training on the [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), and the initial weights of its backbone ResNet50 are pre-trained on [ImageNet](https://image-net.org/). `bird-res50-biAtt-PRE.mat` is also the CRA-CNN for training on the CUB-200-2011 dataset, and the initial weights of its backbone ResNet50 are pre-trained on CUB-200-2011 (ART module, fully-connected classifiers, etc., are randomly initialized). You can choose to load `bird-res50-biAtt.mat` or `bird-res50-biAtt-PRE.mat` to run `train_network.m`. Or you can also download backbones from [MCN-Models](https://www.robots.ox.ac.uk/~albanie/mcn-models.html) and construct CRA-CNN by yourself by referring to `bird-res50-biAtt.mat` or `bird-res50-biAtt-PRE.mat`.

### Citation
 
Please cite our paper if you use CRA-CNN in your work.
```
@inproceedings{liu2020contrastively,
  title={Contrastively-reinforced Attention Convolutional Neural Network for Fine-grained Image Recognition.},
  author={Liu, Dichao and Wang, Yu and Kato, Jien and Mase, Kenji},
  booktitle={BMVC},
  year={2020}
}
```
