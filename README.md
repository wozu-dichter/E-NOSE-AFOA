# Smart electronic nose enabled by an all-feature olfactory algorithm (AFOA)      


### Introduction
This repository contains the code of the paper [Smart electronic nose enabled by an all-feature olfactory algorithm (AFOA)]. Our method combines one-dimensional convolutional and recurrent neural networks with channel and temporal attention modules to fully utilize complementary global and dynamic information in an end-to-end manner. We further demonstrate that a novel data augmentation method can transform the raw data into a suitable representation for feature extraction. The experimental results show that the smart e-nose simply comprising of six semiconductor gas sensors achieves superior performances to state-of-the-art methods on real-world data.

<p align="center"><img width="75%" src="figures/Fig.1.jpg" /></p>

### Prerequisites
* python 3.6
* tensorflow 1.13
* keras 2.2
* numpy
* scikit-learn

### How to Run

**Preparation**.
  1. Modify the dataset path in main.py:

```Shell
tfrecord_file='/home/xxxx/works/E-nose/AFOA/data/
```
  2. Modify the pre-trained model path:

```Shell
pretrained_model_path=/home/xxxx/works/COVID-19/PMP/pretrained_model_dataset1/
```

  3. Copy datasets to your dataset path.
   
   
**Training**.  

For example, train our model on dataset 1

```Shell
python train_on_dataset1.py
```
It will save the models in ```./checkpionts/``` and results in ```./res/```.
   
   
**Testing**.  

For example, directly evaluate the model trained from dataset1 on dataset 2.

```Shell
python test_on_dataset2_baseline.py
```
It will save the results in ```./res/```.
