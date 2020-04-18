Model Architecture: 
ResNet 34, changing the first Con2d’s stride to 1, remove the Max pooling layer, change the Avg pooling layer to get 2*2 output, add a flatten layer right before the final linear layer.
Loss function:
Cross Entropy Loss
Optimizer:
SGD with learning rate = 0.045, momentum=0.9, weight_decay=0.0001, nesterov=True
StepLR with step_size = 1, gamma = 0.98
Number of Epoch:
40
Interesting part: 
The accuracy of validation set improved a lot when reaching 32 epochs.
