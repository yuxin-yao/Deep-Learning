# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)
        """
       
    
        self.x = x

        self.mean = np.mean(self.x,axis=0)
        self.var = np.var(self.x,axis=0)
        
        # Update running batch statistics
        # if ((self.running_mean == np.zeros((1, x.shape[1]))).all()):
        #     self.running_mean = self.mean
        #     self.running_var = self.var
        
        
        
        
            
        if eval:
            
            
            self.norm = (x-self.running_mean)/np.sqrt(self.running_var+self.eps)
            self.out = self.gamma*self.norm+self.beta
            
            
            
        else:
            self.running_mean = self.alpha*self.running_mean+(1-self.alpha)*self.mean
            self.running_var = self.alpha*self.running_var+(1-self.alpha)*self.var
            self.norm = (x-self.mean)/np.sqrt(self.var+self.eps)
            self.out = self.gamma*self.norm+self.beta
        
            
        
        
        return self.out

        
        
        


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        
        #self.dbeta = (sum(delta)/len(self.x)).reshape(1,len(sum(delta)/len(self.x)))
        self.dbeta = (np.array(np.sum(delta,axis=0))).reshape(self.dbeta.shape)
        self.dgamma = np.sum(delta*self.norm,axis = 0).reshape(self.dgamma.shape)
        
        dldxhat = delta*self.gamma
        dldsigma = -1/2*np.sum(dldxhat*(self.x-self.mean)*((self.var+self.eps)**(-3/2)),axis=0)
        dldmu = -np.sum(dldxhat*(self.var+self.eps)**(-1/2),axis=0)-2/self.x.shape[0]*dldsigma*np.sum(self.x-self.mean,axis=0)
        part1 = dldxhat*((self.var+self.eps)**(-1/2))
        part2 = dldsigma*(2/self.x.shape[0]*(self.x-self.mean))
        part3 = dldmu*1/self.x.shape[0]
    
    
        #part1 = delta*self.gamma*((self.var+self.eps)**(-1/2))
        #part2 = -np.mean(delta*self.gamma*(self.x-self.mean)*((self.var+self.eps)**(-3/2)),axis=0)*(self.x-self.mean)
        #part3 = -np.mean(delta*self.gamma*((self.var+self.eps)**(-1/2)),axis=0)
    
        return part1+part2+part3
