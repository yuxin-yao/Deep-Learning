# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import math

class Linear():
    def __init__(self, in_feature, out_feature, weight_init_fn, bias_init_fn):

        """
        Argument:
            W (np.array): (in feature, out feature)
            dW (np.array): (in feature, out feature)
            momentum_W (np.array): (in feature, out feature)

            b (np.array): (1, out feature)
            db (np.array): (1, out feature)
            momentum_B (np.array): (1, out feature)
        """

        self.W = weight_init_fn(in_feature, out_feature)
        self.b = bias_init_fn(out_feature)

        # TODO: Complete these but do not change the names.
        self.dW = np.zeros((in_feature,out_feature))
        self.db = np.zeros((1,out_feature))

        self.momentum_W = np.zeros((in_feature,out_feature))
        self.momentum_b = np.zeros((1,out_feature))
        
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.x = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, out feature)
        """
        self.x = x
        return np.dot(x,self.W)+self.b
        

    def backward(self, delta):

        """
        Argument:
            delta (np.array): (batch size, out feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        self.db = (sum(delta)/len(self.x)).reshape(1,len(sum(delta)/len(self.x)))
        self.dW = np.dot(self.x.T,delta)/len(self.x)
        
        
        return np.dot(delta,self.W.T)
        
        
