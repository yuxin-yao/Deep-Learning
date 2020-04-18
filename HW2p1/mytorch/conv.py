# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)
        
        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
        self.input_size = 0
        self.x = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        output_size = (x.shape[-1]-self.kernel_size)//self.stride + 1
        out = np.zeros((x.shape[0], self.out_channel, output_size))
        self.input_size = x.shape[-1]
        self.x = x

        
        for n in range(x.shape[0]):
            for j in range(self.out_channel):
                for i in range(0, x.shape[-1]-self.kernel_size+1, self.stride):
                    x_segment = x[n, :,i:(i+self.kernel_size)]
                    out[n, j, int(i/self.stride)] = np.sum(np.multiply(self.W[j], x_segment))+self.b[j]
              
        
        return out



    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        dx = np.zeros((delta.shape[0], self.in_channel, self.input_size))

        for n in range(delta.shape[0]):
            for j in range(self.out_channel):
                self.db[j] += np.sum(delta[n, j])
                for i in range(delta.shape[-1]):
                    x_segment = self.x[n, :,i*self.stride:(i*self.stride+self.kernel_size)]
                    self.dW[j] += delta[n, j, i]*x_segment
                    dx[n, :,i*self.stride:(i*self.stride+self.kernel_size)] += delta[n, j, i]*self.W[j]
                    
        return dx



class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        return x.reshape(self.b, self.c*self.w)

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        return delta.reshape(self.b, self.c, self.w)
