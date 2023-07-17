# ML-2020/21: RoboCup@Home Object Classification

This is the second homework of the [ML 2020/21 course](https://sites.google.com/diag.uniroma1.it/machine-learning/home2021) at Sapienza University of Rome.

## Task Assignment

The homework aims at solving an *Image Classification problem* with objects typically available in a home environment. 

## Dataset

A subset of the [RoboCup@Home-Objects dataset](https://sites.google.com/diag.uniroma1.it/robocupathome-objects/home) that has been developed within the [RoboCup@Home](https://athome.robocup.org) competition is represented by 8 zipped folders with images corresponding to the categories listed below:

```
containers/basket_container
tableware/snack_bowl
drinks/soft_drink_bottle
cutlery/dinnerware
fruits/Papayas
snacks/Crackers
food/Oatmeal_box
cleaning_stuff/Upholstery_Cleaners
```

## Approach

The idea was starting from scratch with simple "toy" CNN LeNet first to obtain the best possible with performance and then to use the resultant model as a baseline for comparison with stronger architectures that exploit the concept of *Transfer Learning* with the subsequent fine-tuning. According to the plan, several models were subject to comparative analysis:

1. LeNet
2. VGG16
3. AlexNet
4. ResNet50
5. GoogleNet

Final model uses Transfer Learning with **GoogleNet** (```keras.applications.inception_v3.InceptionV3```) **trained on ImageNet** dataset. It's weights are saved but not uploaded to the current repository for the sake of avoidance of the high memory consumption.

A detailed description of the implemented solution: how data have been preprocessed, which methods/algorithms have been used, which configurations of the method have been tried, description of the evaluation method used, and obtained results using appropriate metrics can be found in the [report.pdf](https://github.com/olga-sorokoletova/Machine-Learning/blob/main/Homework%202/report.pdf).

## Implementation

The Jupiter Notebook with the Tensorflow implementation of the aforementioned models is attached as [code.ipynb](https://github.com/olga-sorokoletova/Machine-Learning/blob/main/Homework%202/code.ipynb).

## Results

The best result achieves:
* ~ 0.9 value of Accuracy metrics
* 0.85 or larger F1-score per class

