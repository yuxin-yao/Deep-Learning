"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')

from linear import Linear
from batchnorm import BatchNorm
from loss import SoftmaxCrossEntropy
from activation import *



class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
        #self.linear_layers = []
        #self.linear_layers.append(Linear(input_size, output_size, weight_init_fn, bias_init_fn))
        
        input_size_list = [self.input_size]+hiddens
        output_size_list = hiddens+[self.output_size]
        weight_init_list = np.repeat(weight_init_fn,self.nlayers)
        bias_init_list = np.repeat(bias_init_fn,self.nlayers)
        para_list = list(zip(input_size_list,output_size_list,weight_init_list,bias_init_list))
        self.linear_layers = [Linear(para_list[i][0],para_list[i][1],para_list[i][2],para_list[i][3]) for i in range(self.nlayers)]
        
        
        
        
        
    

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        self.bn_layers = []
        if self.bn:
            #self.bn_layers.append(BatchNorm(input_size, alpha=0.9))
            self.bn_layers = [BatchNorm(hiddens[i], alpha=0.9) for i in range(num_bn_layers)]
            


    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP.
        
    
        output_itr = x
         
                  
        for i in range(len(self.linear_layers)):
            if i < len(self.bn_layers):

                output_itr = self.activations[i].forward(self.bn_layers[i].forward(self.linear_layers[i].forward(output_itr),not self.train_mode))
            else:
                output_itr = self.activations[i].forward(self.linear_layers[i].forward(output_itr))
        self.output = output_itr
        return output_itr
        #input size, output size, hiddens, activations,
        #weight init fn, bias init fn, criterion.
        #The activation function will be a single Identity activation, and the criterion is a SoftmaxCrossEntropy
        #object.
        #raise NotImplemented

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        for i in range(len(self.linear_layers)):
            # Update weights and biases here
            
            self.linear_layers[i].dW.fill(0.0)
            self.linear_layers[i].db.fill(0.0)
            
        # Do the same for batchnorm layers
        for i in range(len(self.bn_layers)):
            # Update weights and biases here
            
            self.bn_layers[i].dgamma.fill(0.0)
            self.bn_layers[i].dbeta.fill(0.0)

        
        #raise NotImplemented

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)

        for i in range(len(self.linear_layers)):
            # Update weights and biases here
           
            self.linear_layers[i].momentum_W = self.momentum*self.linear_layers[i].momentum_W - self.lr*self.linear_layers[i].dW
            self.linear_layers[i].momentum_b = self.momentum*self.linear_layers[i].momentum_b - self.lr*self.linear_layers[i].db
            #self.linear_layers[i].W = self.linear_layers[i].W - self.lr*self.linear_layers[i].dW
            #self.linear_layers[i].b = self.linear_layers[i].b - self.lr*self.linear_layers[i].db
            
            self.linear_layers[i].W = self.linear_layers[i].W + self.linear_layers[i].momentum_W
            self.linear_layers[i].b = self.linear_layers[i].b + self.linear_layers[i].momentum_b
            
        # Do the same for batchnorm layers
        for i in range(len(self.bn_layers)):
            # Update weights and biases here
            
            self.bn_layers[i].gamma = self.bn_layers[i].gamma - self.lr*self.bn_layers[i].dgamma
            self.bn_layers[i].beta = self.bn_layers[i].beta - self.lr*self.bn_layers[i].dbeta

        #raise NotImplemented

    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        #loss = SoftmaxCrossEntropy()
        # self.criterion.forward(self.activations[0].state,labels)
        # delta = self.criterion.derivative()
        

        # return self.linear_layers[0].backward(delta)
        if not self.bn:
            self.criterion.forward(self.activations[len(self.linear_layers)-1].state,labels)
            delta = self.criterion.derivative()
            for i in range(len(self.linear_layers)-1,-1,-1):
                
                delta = self.linear_layers[i].backward(self.activations[i].derivative()*delta)
        
        else:
            self.criterion.forward(self.activations[len(self.linear_layers)-1].state,labels)
            delta = self.criterion.derivative()
           
            for i in range(len(self.linear_layers)-1,-1,-1):
                if i < len(self.bn_layers):
                    delta = self.linear_layers[i].backward(self.bn_layers[i].backward(self.activations[i].derivative()*delta)) #1
                else:
                    delta = self.linear_layers[i].backward(self.activations[i].derivative()*delta) #2
            
        return delta
        
       
        
    

    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...
    mlp.train()

    for e in range(nepochs):

        # Per epoch setup ...
        loss = 0
        num = 0
        error=0

        

        for b in range(0, len(trainx), batch_size):
            order = np.random.permutation(batch_size)
            #trainx_epoch_data,trainy_epoch_data = np.random.shuffle(trainx[b:b+batch_size],trainy[b:b+batch_size])
            trainx_epoch_data = (trainx[b:b+batch_size])[order]
            trainy_epoch_data = (trainy[b:b+batch_size])[order]
          
            mlp.zero_grads()
            mlp.forward(trainx_epoch_data)
            mlp.backward(trainy_epoch_data)
            mlp.step()
            num += 1
            
            
            
            loss += mlp.total_loss(trainy_epoch_data)
            
            error += mlp.error(trainy_epoch_data)/batch_size
        training_errors[e]  =error/(len(trainx)/batch_size)
        training_losses[e] = loss/(len(trainx))
       
            
            # Train ...
        num = 0
        loss = 0
        error=0
        mlp.eval()
        for b in range(0, len(valx), batch_size):
            order = np.random.permutation(batch_size)
            valx_epoch_data = (valx[b:b+batch_size])[order]
            valy_epoch_data = (valy[b:b+batch_size])[order]
            
            
            num +=1
            mlp.forward(valx_epoch_data)
            loss += mlp.total_loss(valy_epoch_data)
            error += mlp.error(valy_epoch_data)/batch_size
        validation_errors[e]  =error/(len(valx)/batch_size)
        validation_losses[e] = loss/(len(valx))
            
            # Val ...
            

        # Accumulate data...

    # Cleanup ...

    # Return results ...
        

    return (training_losses, training_errors, validation_losses, validation_errors)

    
