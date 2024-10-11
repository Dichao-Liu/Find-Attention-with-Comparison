# Find ttention with Comparison

The proposed framework is inspired by human behavior: (1) humans often use knowledge from one task to assist in learning another related task; (2) humans instinctively compare images from the same or different categories to find commonalities and differences. This framework is designed to simulate these cognitive processes. Extensive experimental results demonstrate that the framework significantly improves fine-grained image recognition accuracy across various backbone network architectures and public datasets.

For more details, you can refer to our [BMVC paper](https://www.bmvc2020-conference.com/conference/papers/paper_0656.html) and [Multimedia Systems paper](https://link.springer.com/article/10.1007/s00530-024-01446-1).

![CRA-CNN Framework](https://github.com/Dichao-Liu/Find-Attention-with-Comparison/blob/main/CRA-CNN.png)


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

**To reproduce the best accuracy** reported in the BMVC paper at the inference phase, please **only resize** and **do not crop** the input images. Then for each input image, you will obtain a tensor whose size is *`h`×`w`×`num_of_classes`* (*`h`* and *`w`* vary between different specific images) at the last fully-connected layer (i.e., the classifier). The **final prediction score** for the input image is computed by averaging the *`h`×`w`×`num_of_classes`* tensor into *`1`×`1`×`num_of_classes`* prediction score.


## Bibtex

```
@article{liu2024mt,
  title={MT-ASM: a multi-task attention strengthening model for fine-grained object recognition},
  author={Liu, Dichao and Wang, Yu and Mase, Kenji and Kato, Jien},
  journal={Multimedia Systems},
  volume={30},
  number={5},
  pages={297},
  year={2024},
  publisher={Springer}
}
```


```
@inproceedings{liu2020contrastively,
  title={Contrastively-reinforced Attention Convolutional Neural Network for Fine-grained Image Recognition.},
  author={Liu, Dichao and Wang, Yu and Kato, Jien and Mase, Kenji},
  booktitle={BMVC},
  year={2020}
}
```


