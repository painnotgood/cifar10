# CIFAR-10 ResNet Image Classfication

# 1. Introduction
In recent years, image classification has been a prevalent direction of research and application. This Jupyter Notebook will utilize Residual Network (ResNet) and some techniques to aim to perform with maximum accuracy while having a fast training time.

## Goals:
- Aim for 94+% accuracy (the error rate of a human on CIFAR-10 was estimated to be 6% => 94+% accuracy = human-level performance)
- Train a model, not using pretrained models, under 3 hours using Google Colab's L4 High-RAM GPU

# 2. Execution

Libraries: torch, torchvision, numpy
 
## Steps
1. Import needed libraries.
2. Initialize hyperparameters and set the device to utilize NVIDIA's CUDA:
   - batch_size = 64 for accurate results while keeping the process fast
   - num_epochs = 10 at first to tune learning rates (I reduced it to 5 later for quicker tuning time)
   - lr = 0.1 (this is the most important hyperparameter that should be tuned first)
   - num_classes = 10, corresponding to CIFAR10's 10 classes
3. Import the dataset with augmentations to further improve model's performance.
4. Define the model:
   - set the seed for reproducible results
   - split the training set into validation and training subsets (10% for validation since the training size is small)
   - load resnet34 architecture without the weights; the architecture can be seen here: https://www.researchgate.net/figure/Resnet-34-Architecture_fig1_354122133
   - resize the fully connected layers to the number of classes of CIFAR-10
   - replace conv1 into a smaller kernel with small steps and only 1 layer of padding (CIFAR-10 images are much smaller than ImageNet, 32x32 and 224x224 respectively, therefore it is necessary to resize the convolutional layers)
   - remove the maxpool layer (to keep the resolution higher as it is small already)
   - use stochastic gradient descent as optimizer, cosine annealing LR as scheduler, and cross entropy loss as criterion
   - the model will stop if the improvements are smaller than delta or larger than the prior run for 10 epochs (in another word, the model is not improving anymore)
   Note: using scaler and autocast to CUDA improve the model efficiency even further
5. Tune the LR (~30-60 minutes):
   - use uniform randomization to generate random learning rates, starting from 10^0 to 10^(-6); then, look for the best ranges and reduce gradually
   - repeatedly get 10 lrs and run for 5 epochs each to get an excellent lr
6. Do an ensemble of 10 models, each trained for 100 epochs, and get the average of the ensemble

# 3. Result
In my run, I got 95.09% accuracy for the model ensemble in 2.5 hours, which checks all the goals.
It can be observed that lots of the model runs stop even before 50 epochs, which saved the time a lot. If the runs had been longer, it might break the time limit.
   
