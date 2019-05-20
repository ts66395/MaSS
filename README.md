# MaSS
Momentum-added Stochastic Solver (MaSS): implemented in Tensorflow(Keras)

## Introduction
MaSS (Momentum-added Stochastic Solver) is an accelerated stochastic gradient method for training over-parametrized models.
The code for the algorithm is in the file optimizers.py
## Requirement
    Python: >= 3.5.2
    Tensorflow: >= 1.8.0
    Keras: 2.1.5

## Running Experiments
An example experiment: training a ResNet-32 using MaSS to classify the CIFAR-10 images. Run the code:
    
    $ python3 train.py
  
or (specify CUDA device)

    $ CUDA_VISIBLE_DEVICES=0 python3 train.py
    
Our experiment achieves on average 92.8% classification accuracy on the test set of CIFAR-10.
