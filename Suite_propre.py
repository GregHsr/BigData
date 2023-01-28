import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sys
from IPython.display import display

def verif(bias,n):
    for k in bias:
        if k.shape != (n,1):
            raise ValueError("Le biais n'est pas de la bonne dimension")

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

def relu(x, mode = "n"):
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
    print('x shape:', x.shape)
    print('NpL:', NpL)
    print('Nfx:', Nfx)
    print('Wts:', len(Wts))

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
        print('WL shape:', WL.shape)
        print('yLm1 shape:', yLm1.shape)
        print('bL shape:', bL.shape)
        
        zL = np.dot(WL, yLm1) + np.tile(bL,(n,1)).T
        z.append(zL)

        print('zL shape:', zL.shape)
        print('bL shape:', bL.shape)
        print("n",n)

        ## activation
        ## to call a function given its name, use the function fx = globals()["fun_name"]
        sigma = globals()[ActivFun[iL]]
        yL = sigma(zL)
        y.append(yL)
        ## move to next layer
        yLm1 = yL
    return y,z


###### Task 5: Read data ######

df = pd.read_excel("abalone_data.xlsx", sheet_name="Sheet1")
display(df.head(5).style.format("{0:.2f}").set_caption("Few lines of the dataset :"))
dfsum = datasum(df)
pd.options.display.float_format = '{:,.2f}'.format
display(dfsum.style.hide(axis = "index").set_caption("Statistics of the dataset"))


###### Task 6: Split data ######

## percentage of data to be used for training
percTrain = 0.7

## index of training and test
#trainindex = np.random.rand(len(df)) < percTrain
dftrain = df.sample(frac = percTrain, axis = 0)
dftest = df.drop(dftrain.index)

print("percTrain",len(dftrain)/len(df))
print("percTest",len(dftest)/len(df))

###### Task 7: Defining features and output variables ######

ytrain = dftrain.Age_yr
xtrain = dftrain.drop("Age_yr", axis = 1)

print("xtrain.head" ,xtrain.head())

ytest = dftest.Age_yr
xtest = dftest.drop("Age_yr", axis = 1)

## printing some information
print('Shape of original data : ', df.shape)
print('xtrain : ',xtrain.shape, 'ytrain : ',ytrain.shape)
print('xtest  : ',xtest.shape,  'ytest  : ',ytest.shape)


###### Task 8: Scaling the data ######

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
display(x_train_summary.style.hide(axis = "index").set_caption("Statistics of the dataset - before scaling"))
display(x_train_scl_summary.style.hide(axis = "index").set_caption("Statistics of the dataset - after scaling"))


###### Task 9: Train the neural network using the gradient descent method based on backpropagation ######

def ANN_backpro(x, ytrue, NpL, Nfx, Wts, bias, ActivFun, lr):
    '''
    o Shape of x: Nfx * n, where n is the number of data points, and Nfx the number of features
    o Shape of ytrue: n * 1, where n is the number of data points.
    
    '''
    print("x",x.shape)
    print("ytrue",ytrue.shape)
    
    ## step 1: feed forward
    n = x.shape[1]
    yLm1 = x
    z = []
    y = []
    for iL in np.arange(len(NpL)):
        ## get the parameters for the current layer
        WL = Wts[iL]
        bL = bias[iL]
        ## estimate zL from yLm1
        zL = np.dot(WL, yLm1) + np.tile(bL,(n,1)).T
        z.append(zL)
        print("zL",zL.shape)

        ## estimate yL from zL
        sigma = globals()[ActivFun[iL]]
        print("sigma",sigma)
        yL = sigma(zL, mode = "n")
        y.append(yL)

        ## update yLm1
        yLm1 = yL

    ## step 2: backpropagation
    ytrue = ytrue.T
    dJ_dy = 2*(yL - ytrue)                                          ####### crochets en trop??

    for iL in reversed(np.arange(len(NpL))):
        print("iL",iL)
        ## getting zL of the current layer
        zL = z[iL]

        ## estimating dJ_dz from dJ_dy
        sigma = globals()[ActivFun[iL]]
        dJ_dz = dJ_dy * sigma(zL, mode = "d")

        ## getting the parameters of current layer
        WL = Wts[iL]
        bL = bias[iL]
       
        ## estimating dJ_dW from dJ_dz
        ## dJ_dz : (nL x n)
        ## yLm1 : (nL-1 x n)
        ## getting yL-1
        if iL == 0:
            yLm1 = x
        else:
            yLm1 = y[iL-1]

        dJ_dW = np.dot(dJ_dz, yLm1.T)
        
        print("dJ_dz",dJ_dz.shape, "nL x n")
        print("yLm1",yLm1.shape, "nL-1 x n")

        # estimating dJ_db from dJ_dz
        # J_db : nL x 1
        # dJ_dz : nL x n
        dJ_db = np.dot(dJ_dz, np.ones((n,1)))

        print("dJ_db",dJ_db.shape, "nL x 1")
        print("dJ_dz",dJ_dz.shape, "nL x n")

        ## backpropagating the gradient from layer L to layer L-1
        ## WL : nL x nL-1
        ## dJ_dz : nL x n
        ## dJ_dy (L-1) : nL-1 x n
        dJ_dy = np.dot(WL.T, dJ_dz)

        print("WL",WL.shape, "nL x nL-1")
        print("dJ_dz",dJ_dz.shape, "nL x n")
        print("dJ_dy",dJ_dy.shape, "nL-1 x n")
        print("dJ_dW",dJ_dW.shape, "nL x nL-1")
        print("bL",bL.shape, "nL x 1","avant")
        ## Updating the parameters
        print(type(np.multiply(dJ_db,lr)))

        WL = WL - np.multiply(dJ_dW,lr)
        bL = bL - np.multiply(dJ_db,lr).T
        print("WL",WL.shape, "nL x nL-1")
        print("bL",bL.shape, "nL x 1","apres")
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
ActivFun = ['logistic', 'relu']

## List of size of layers -1: important for automatic definition of parameters
NpLm1 = [Nfx]
for iL in np.arange(len(NpL)-1):
    NpLm1.append(NpL[iL])
print("Number of neurons per layer: ", NpL)
print("Number of neurons from the previous layer: ", NpLm1)

## Learning rate
lr = 0.001

## Number of epochs
epochs = 10

## List of weights and biases
Wts = []
bias = []
for iL in np.arange(len(NpL)):
    ## random initialization
    WL = (np.random.rand(NpL[iL], NpLm1[iL])-0.5)*2
    bL = (np.random.rand(NpL[iL])-0.5)*2
    ## appending
    Wts.append(WL)
    bias.append(bL)


###### Task 11: keep track of (1) epochs, (2) training error (MSE), and (3) test error ######

print("Learning rate: ", lr, "  Number of epochs: ", epochs)
MSEtrain = np.array([])
MSEtest = np.array([])
epoch = np.array([])
sys.stdout.write('\r')

for iepoch in np.arange(epochs):
    epoch = np.append(epoch, iepoch)
    ## train the neural network
    x_in = xtrain_scl.T 
    Wts, bias = ANN_backpro(x = x_in, ytrue = ytrain, NpL = NpL, Nfx = Nfx, Wts = Wts, bias = bias, ActivFun = ActivFun, lr = lr)

    print("Wts[0]", Wts[0].shape)
    print("Wts[1]", Wts[1].shape)

    ## estimate the MSE for the train dataset
    x_in = xtrain_scl.T
    print("x_in",x_in.shape)
    yout, zout = ANN(x = x_in, NpL = NpL, Nfx = Nfx, Wts = Wts, bias = bias, ActivFun = ActivFun)
    ytrainsim = yout
    print(yout)
    ntrain = ytrain.shape[0]
    print("ytrainsim",len(ytrainsim))
    print("ytrain",ytrain.shape)
    Error_train = np.dot((ytrain - ytrainsim[0]).T,(ytrain - ytrainsim[0]))/ntrain
    
    ## estimate the MSE for the test dataset
    x_in = xtest_scl.T
    yout, zout = ANN(x = x_in, NpL = NpL, Nfx = Nfx, Wts = Wts, bias = bias, ActivFun = ActivFun)
    ytestsim = yout
    ntest = ytest.shape[0]
    Error_test = np.dot((ytest - ytestsim[0]).T,(ytest - ytestsim[0]))/ntest
   
    ## keeping track of the errors
    MSEtrain = np.append(MSEtrain, Error_train[0,0])
    MSEtest = np.append(MSEtest, Error_test[0,0]) 
    
    ## print the evolution
    sys.stdout.write('\r' "Epoch: " + str(int(iepoch + 1)).rjust(5,'0') + "/"
                    + str(int(epochs)).rjust(5,'0') + " " +
                    "Training error: " + str(round(Error_train[0,0],2)) + 
                    "  Test error: " + str(round(Error_test[0,0],2)))

## plot the evolution of the errors

## initialization of the plot
plt.grid(color='black', axis='y', linestyle='-', linewidth=0.5)    
plt.grid(color='black', axis='x', linestyle='-', linewidth=0.5)   
plt.grid(which='minor',color='grey', axis='x', linestyle=':', linewidth=0.5)     
plt.grid(which='minor',color='grey', axis='y', linestyle=':', linewidth=0.5)    
plt.xticks(fontsize=16); plt.yticks(fontsize=16)   
plt.xlabel('epoch',fontsize=16 )
plt.ylabel(r'$RMSE_{train}$ (yr), $RMSE_{test}$ (yr)', size = 16)
## plotting the data
plt.plot(epoch, MSEtrain**0.5, color = "blue", linewidth = 2., label = "Training error")
plt.plot(epoch, MSEtest**0.5, color = "orange", linewidth = 2., label = "Test error")
plt.title("Prediction error", fontsize = 16)
plt.gcf().set_size_inches(10, 5)
plt.legend(loc="upper right", prop={'size': 15})
plt.savefig("fig01.png", dpi = 300,  bbox_inches='tight')
plt.show()

####### Task 12: Use optimized parameters #######

## Showing the results for the test dataset
x_in = xtest_scl.T
yout, zout = ANN(x = x_in, NpL = NpL, Nfx = Nfx, Wts = Wts, bias = bias, ActivFun = ActivFun)
ytestsim = yout
Error_test = np.dot((ytest - ytestsim[0]).T,(ytest - ytestsim[0]))/ntest
## Making a scatter plot
## initialization of the plot
plt.grid(color='black', axis='y', linestyle='-', linewidth=0.5)    
plt.grid(color='black', axis='x', linestyle='-', linewidth=0.5)   
plt.grid(which='minor',color='grey', axis='x', linestyle=':', linewidth=0.5)     
plt.grid(which='minor',color='grey', axis='y', linestyle=':', linewidth=0.5)    
plt.xticks(fontsize=16); plt.yticks(fontsize=16)   
plt.xlabel(r'$Age_{obs}$ (yr)',fontsize=16 )
plt.ylabel(r'$Age_{sim}$ (yr)',fontsize=16 )
## plotting the data
plt.scatter(ytest, ytestsim, color = "red", marker = "o")
plt.plot([0., 30.], [0., 30.], color='k', linestyle='-', linewidth=2)
plt.gcf().set_size_inches(6, 6)
plt.savefig("fig02.png", dpi = 300,  bbox_inches='tight')
plt.show()
