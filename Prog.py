import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sys


###### Task 1: Define the function ######

## a function that takes a dataframe and returns some basic stastics of its columns
def datasum(df):
    dfmin = df.apply(np.min, 0)
    dfmedian = df.apply(np.median, 0)
    dfmean = df.apply(np.mean,0)
    dfmax = df.apply(np.max, 0)
    dfstd = df.apply(np.std, 0)
    dfsummary = pd.DataFrame(np.array([dfmean, dfstd, dfmin, dfmedian, dfmax]), columns=list(df.columns[:]))
    dfsummary["Statistic"] = ["mean", "std", "min", "median", "max"]
    xcol = dfsummary.shape[1]
    colarr = [dfsummary.columns[xcol-1]]
    for item in dfsummary.columns[0:(xcol-1)]:
        colarr.append(item)
    dfto = dfsummary[colarr]
    return dfto

## activation functions and their derivatives
def logistic(x, mode = "n"):
    '''
    mode = "n" : returns f(x)
    mode = "d": returns f'(x)
    '''
    t = np.exp(-x)
    if mode == "n":
        fx = 1/(1+t)
    if mode == "d":
        fx = t/(1+t)**2
    return fx

## define new functions here
def tanh(x, mode = "n"):
    '''
    mode = "n" : returns f(x)
    mode = "d": returns f'(x)
    '''
    p = np.exp(x)
    n = np.exp(-x)
    if mode == "n":
        fx = p-n/(p+n)
    if mode == "d":
        fx = 1- (p-n/(p+n))**2
    return fx

def ReLU(x, mode = "n"):
    '''
    mode = "n" : returns f(x)
    mode = "d": returns f'(x)
    '''
    if mode == "n":
        fx = np.where(x < 0, 0, x)
    if mode == "d":
        fx = np.where(x < 0, 0, x)
        fx = np.where(x >= 0, 1, x)
    return fx

def Linear(x, mode = "n"):
    '''
    mode = "n" : returns f(x)
    mode = "d": returns f'(x)
    '''
    if mode == "n":
        fx = x
    if mode == "d":
        fx = 1
    return fx


## NL is the number of layers including the hidden layers AND the output layer
NL = 2
## NpL defines the number of neurones per each layer. The length of NpL should be exactly equal to NL
NpL = [3, 1]
## ActivFun defines the activation functions per each layer
ActivFun = ['logistic', 'logistic']

## Add the number of features x
Nfx = 2
## List of size of layers -1: important for automatic definition of parameters
NpLm1 = [Nfx]
for iL in np.arange(len(NpL)-1):
    NpLm1.append(NpL[iL])
print("Number of neurons per layer: ", NpL)
print("Number of neurons from the previous layer: ", NpLm1)

###### Task 2: Define the parameters ######

## List of weights and biases
Wts = []
bias = []
for iL in np.arange(len(NpL)):
    ## random initialization
    ## the initial parameters should be between -1 and 1
    ## use the function np.random.rand to make random initializations (but these will be between 0 and 1)
    WL = (np.random.rand(NpL[iL], NpLm1[iL])-0.5)*2
    print(WL)
    bL = (np.random.rand(NpL[iL])-0.5)*2
    ## appending
    Wts.append(WL)
    bias.append(bL)
print('Wts[0].shape',Wts[0].shape)
print('bias[0].shape',bias[0].shape)

###### Task 3: Define the ANN function ######

def ANN(x, NpL, Nfx, Wts, bias, ActivFun):
    '''
    This function computes the output of a neural network containing NL layers
    where NL is the length of NpL. 
    o NpL contains the number of neurons per each hidden layer + the output layer. 
    o Nfx is the number of input features.
    o Wts is a list that contains the 2D arrays of weights for each layer.
    Each 2D array of weights has dimensions nL x nL-1, nL being the number of neurons
    of current layer L, and nL-1 the number of neurons (or features) of layer L-1.
    o bias is a list of 1D arrays of weights for each layer.
    Each 1D array has dimensions nL x 1. When there are many data points (or members), say
    n data points, the bias should be repeated using the function np.tile.
    o ActivFun contains the name of the activation function for each layer.
    
    '''
    n = x.shape[1]
    yLm1 = x

    print('n vaut:', n)
    print('NL vaut:', len(NpL))
    
    ## z is a list that saves the zL arrays for each hidden and output layer
    z = []
    ## similarly, y is a list that saves the yL arrays. Specifically, yL[NL-1] contains the output.
    y = []
    for iL in np.arange(len(NpL)):
        ## parameters
        WL = Wts[iL]
        bL = bias[iL]
        ## multiplication
        ## make sure that the operation is correct for n individual points.
        ## the dimension of zL should be nL x n. Since bL is nL x 1, use np.tile to overcome this issue.
        
        print('valeurs',WL,yLm1)

        zL = np.dot(WL, yLm1) + np.tile(bL, (n,1)).T
        z.append(zL)
        print('zL',zL.shape)
        print("bL",bL.shape)
        print("n",n)

        ## activation
        ## to call a function given its name, use the function fx = globals()["fun_name"]
        sigma = globals()[ActivFun[iL]]
        yL = sigma(zL)
        y.append(yL)
        ## move to next layer
        yLm1 = yL
    return y,z


## Input: 4 data points x 2 features (Nfx = 2)
input_features = np.array([[0,0], 
                           [0,1], 
                           [1,0], 
                           [1,1]])
print(input_features.shape)
print(input_features)

# Output: 4 data points x 1 feature
target_output = np.array([[0], [1], [1], [1]])
print(target_output.shape)
print(target_output)

## This where you can test your neural network
x_in = input_features.T

#print(x_in)
#print(NpL)
#print(Nfx)
#print(bias[0])
#print(ActivFun)
#print(Wts[0])

y,z = ANN(x = x_in, NpL = NpL, Nfx = Nfx, Wts = Wts, bias = bias, ActivFun = ActivFun)
ysim = y[NL-1].T
print("ysim.shape", ysim.shape)

###### Task 4: Compute the MSE and RMSE ######

n = target_output.shape[0] 

MSE = np.dot((target_output - ysim).T,(target_output - ysim))/n
print("MSE = ", MSE)

RMSE = np.sqrt(MSE)
print("RMSE = ", RMSE)

###### Task 5: Read data ######

df = pd.read_excel("abalone_data.xlsx", sheet_name="Sheet1")
print(df.head(5).style.format("{0:.2f}").set_caption("Few lines of the dataset :"))
dfsum = datasum(df)
pd.options.display.float_format = '{:,.2f}'.format
print(dfsum.style.hide(axis = "index").set_caption("Statistics of the dataset"))

###### Task 6: Split data ######

## percentage of data to be used for training
percTrain = 0.7

## index of training and test
#trainindex = np.random.rand(len(df)) < percTrain
dftrain = df.sample(frac=percTrain, axis = 0)
dftest = df.drop(dftrain.index)
print(len(dftrain)/len(df))
print(len(dftest)/len(df))

###### Task 7: Defining features and ouput variables ######

ytrain = dftrain.Age_yr
xtrain = dftrain.drop(['Age_yr'], axis = 1)
print(xtrain.head())
print(ytrain.shape)
ytest = dftest.Age_yr
xtest = dftest.drop(['Age_yr'], axis = 1)

## printing some information
print('Shape of original data : ', df.shape)
print('xtrain : ',xtrain.shape, 'ytrain : ',ytrain.shape)
print('xtest  : ',xtest.shape,  'ytest  : ',ytest.shape)

###### Task 8: Scaling data ######

## estimate the mean and the standard deviation from the train dataset
xmean = xtrain.mean()
xstd = xtrain.std()

## scaling
xtrain_scl = (xtrain - xmean)/xstd
xtest_scl = (xtest - xmean)/xstd

x_train_summary = datasum(xtrain)
x_test_summary = datasum(xtest)
x_train_scl_summary = datasum(xtrain_scl)
x_test_scl_summary = datasum(xtest_scl)
## convert to arrays
xtrain_scl, ytrain = np.array(xtrain_scl), np.array(ytrain)
xtest_scl, ytest = np.array(xtest_scl), np.array(ytest)
## print the dataset before and after scaling
print(x_train_summary.style.hide(axis = "index").set_caption("Statistics of the dataset - before scaling"))
print(x_train_scl_summary.style.hide(axis = "index").set_caption("Statistics of the dataset - after scaling"))

###### Task 9: Train the neural network using the gradient descent method based on backpropagation ######

def ANN_backpro(x, ytrue, NpL, Nfx, Wts, bias, ActivFun, lr):
    '''
    o Shape of x: Nfx * n, where n is the number of data points, and Nfx the number of features
    o Shape of ytrue: n * 1, where n is the number of data points.
    '''
    ## step 1: feed forward
    n = x.shape[1]
    print("n", n)
    print("ytrue.shape", ytrue.shape)
    yLm1 = x
    z = []
    y = []
    for iL in np.arange(len(NpL)):
        ## get the parameters for the current layer
        WL = Wts[iL]
        bL = bias[iL]
        ## estimate zL from yLm1
        zL = np.dot(WL,yLm1) + np.tile(bL, (n,1)).T
        z.append(zL)

        ## activation: estimate yL from zL
        sigma = globals()[ActivFun[iL]]
        yL = sigma(zL)
        y.append(yL)

        ## move to next layer
        yLm1 = yL

    ## step 2: backpropagation
    print("ytrue.shape", ytrue.shape)
    ytrue = ytrue.T
    print("ytrue.T.shape", ytrue.shape)
    print("yL.shape", yL.shape)
    dJ_dy = 2*(yL - ytrue) 
    print("dJ_dy.shape", dJ_dy.shape)

    for iL in reversed(np.arange(len(NpL))):
        ## getting zL of the current layer
        zL = z[iL]
        ## estimating dJ_dz from dJ_dy
        sigma = globals()[ActivFun[iL]]
        dJ_dz = dJ_dy * sigma(zL,"d")
        ## getting the parameters of current layer
        WL = Wts[iL]
        bL = bias[iL]
        
        ## estimating dJ_dW from dJ_dz
        ## dJ_dz : (nL x n)
        ## yLm1 : (nL-1 x n)
        ## getting yL-1
        if(iL == 0):
            yLm1 = x
        else:
            yLm1 = y[iL-1] 
        dJ_dW = np.dot(dJ_dz, yLm1)
        
        ## estimating dJ_db from dJ_dz
        ## dJ_db : nL x 1
        ## dJ_dz : nL x n
        dJ_db = np.sum(dJ_dz, axis = 1, keepdims = True)  ####### FAUXXXXXXX
        
        ## backpropagating the gradient from layer L to layer L-1
        ## WL : nL x nL-1
        ## dJ_dz : nL x n
        ## dJ_dy (L-1) : nL-1 x n
        dJ_dy = np.dot(WL, dJ_dz)
        
        ## Updating the parameters
        WL = WL - lr*dJ_dW
        bL = bL - lr*dJ_db
        Wts[iL] = WL 
        bias[iL] = bL
    
    return Wts, bias

###### Task 10: Initialize the parameters of the neural network ######

## NL is the number of layers including the hidden layers AND the output layer
NL = 2
## NpL defines the number of neurones per each layer. The length of NpL should be exactly equal to NL
NpL = [6,1]
## Add the number of features x
Nfx = 8
## ActivFun defines the activation functions per each layer
ActivFun = ['logistic', 'ReLU']

## List of size of layers -1: important for automatic definition of parameters
NpLm1 = [Nfx]
for iL in np.arange(len(NpL)-1):
    NpLm1.append(NpL[iL])
print("Number of neurons per layer: ", NpL)
print("Number of neurons from the previous layer: ", NpLm1)

## Learning rate
lr = 0.001

## Number of epochs
epochs = 5000

## List of weights and biases
Wts = []
bias = []
for iL in np.arange(len(NpL)):
    ## random initialization
    WL = (np.random.rand(NpL[iL], NpLm1[iL])-0.5)*2
    print(WL)
    bL = (np.random.rand(NpL[iL])-0.5)*2
    ## appending
    Wts.append(WL)
    bias.append(bL)

###### Task 11: Train the neural network ######

print("Learning rate: ", lr, "  Number of epochs: ", epochs)
MSEtrain = np.array([])
MSEtest = np.array([])
epoch = np.array([])
sys.stdout.write('\r')
for iepoch in np.arange(epochs):
    epoch = np.append(epoch, iepoch)
    ## train the neural network
    x_in = (df.drop(['Age_yr'], axis = 1)).T
    print("x_in.shape", x_in.shape)
    Wts, bias = ANN_backpro(x = x_in, ytrue = ytrain, NpL = NpL, Nfx = Nfx, Wts = Wts, bias = bias, ActivFun = ActivFun, lr = lr)
        
    ## estimate the MSE for the train dataset
    x_in = xtrain_scl
    yout, zout = ANN(x = x_in, NpL = NpL, Nfx = Nfx, Wts = Wts, bias = bias, ActivFun = ActivFun)
    ytrainsim = yout
    ntrain = ytrain.shape[0]
    Error_train = np.sum((ytrainsim - ytrain)**2)/ntrain
    
    ## estimate the MSE for the test dataset
    x_in = xtest_scl
    yout, zout = ANN(x = x_in, NpL = NpL, Nfx = Nfx, Wts = Wts, bias = bias, ActivFun = ActivFun)
    ytestsim = yout
    ntest = ytest.shape[0]
    Error_test = np.sum((ytestsim - ytest)**2)/ntest
   
    ## keeping track of the errors
    MSEtrain = np.append(MSEtrain, Error_train[0,0])
    MSEtest = np.append(MSEtest, Error_test[0,0]) 
    
    ## print the evolution
    sys.stdout.write('\r' "Epoch: " + str(int(iepoch + 1)).rjust(5,'0') + "/"
                    + str(int(epochs)).rjust(5,'0') + " " +
                    "Training error: " + str(round(Error_train[0,0],2)) + 
                    "  Test error: " + str(round(Error_test[0,0],2)))

