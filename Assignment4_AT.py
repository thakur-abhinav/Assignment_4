import struct as st
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def predict(w1,w2,b1,b2,x):
    '''
    This function predicts the value of y for a given input
    '''
    z1=np.dot(w1,x)+b1
    a1=act(z1,'lrelu')
    a1=batch_norm(a1,1)
    z2=np.dot(w2,a1)+b2
    a2=act(z2,'sigmoid')
    return a2
    
def act(x,k):
    '''
    This function returns the array after applying 
    the specified activation function to each element 
    of the array. k=lrelu,sigmoid
    '''
    if k=='lrelu':
        return np.where(x > 0, x, x * 0.01)
    elif k=='sigmoid':
        return 1/(1+np.exp(-x))
    

def batch_norm(X,k):
    '''
    This function normalises a 2D matrix to mean = 0 and
    std deviation = 1
    '''
    if k==1:
        x = (X - X.mean(axis=0)) / X.std(axis=0)
        return x
    elif k==0:
        x = (X - X.min(axis=0)) / (X.max(axis=0)-X.min(axis=0))
        return x
        
def cel(y_pred,y_true):
    '''
    This function returns the cross entropy loss of
    predicted values wrt true values.
    y_pred=[nx10],y_true=[nx10]
    '''
    return np.sum(y_true*(-(np.log(y_pred))))

def sse(y_pred,y_true):
    '''
    This function returns the sum squared errors of
    predicted values wrt true values.
    y_pred=[nx10],y_true=[nx10]
    '''
    return np.sum((y_true-y_pred)**2)

def accuracy(y_pred,y_true):
    '''
    This function returns the accuracy by deciding the predicted class 
    on the basis of maximum probability out of all 10 class probability
    '''

    a1=np.max(y_pred,axis=0)
    a2=np.max(y_pred*y_true,axis=0)
    k=0
    for i in range(y_true.shape[1]):
        if a1[i]-a2[i]==0:
            k+=1
    return k*100/y_true.shape[1]    


def neural_net(X,Y,x,y,h,b,a,ITE):
    total_batches= Y.shape[1]//b
    w1=2/np.sqrt(784) *np.random.randn(h,784) #xavier initialisation for ReLu
    b1=np.zeros((h,1))
    w2=1/np.sqrt(h)*np.random.randn(10,h) #xavier initialisation for sigmoid
    b2=np.zeros((10,1))
    epoch=0
    while True:
        for i in range(total_batches):
            #defining mini batch and calculating z1,a1,z2,a2
            X1=X[:,i*b:(i+1)*b]
            Y1=Y[:,i*b:(i+1)*b]
            z1=np.dot(w1,X1)+b1
            a1=act(z1,'lrelu')
            a1=batch_norm(a1,1) #batch normalisation for second layer input
            z2=np.dot(w2,a1)+b2
            a2=act(z2,'sigmoid')
            #calculating derivatives for back propogation
            dLda2=(a2-Y1)/b 
            da2dz2=a2*(1-a2)
            dz2dw2=a1
            dz2db2=np.ones((b,1))
            dz2da1=w2
            da1dz1=np.where(a1>0,1,0.01)
            dz1dw1=X1
            dz1db1=np.ones((b,1))
            #back propogation
            dw2=np.dot((dLda2)*(da2dz2),(dz2dw2).T)
            db2=np.dot((dLda2)*(da2dz2),dz2db2)
            dw1=np.dot((np.dot((dz2da1.T),(dLda2)*(da2dz2))*da1dz1),dz1dw1.T)
            db1=np.dot((np.dot((dz2da1.T),(dLda2)*(da2dz2))*da1dz1),dz1db1)
            #parameter update
            w1=w1-a*dw1
            w2=w2-a*dw2
            b1=b1-a*db1
            b2=b2-a*db2
        epoch+=1
        #plotting error vs ite curve for training(blue) and test(red)
        a2=predict(w1,w2,b1,b2,X)
        L_train=sse(a2,Y)
        a2=predict(w1,w2,b1,b2,dev_x)
        L_dev=sse(a2,dev_y)
        plt.scatter(epoch,L_train,color='blue',label='Train data')
        plt.scatter(epoch,L_dev,color='red',label='Test data')
        print("ITE:",epoch,"Train error:",L_train,"Validation error:",L_dev)
        if epoch==ITE:
            plt.xlabel("ITE")
            plt.ylabel("Error(SSE)")
            plt.show()
            return w1,w2,b1,b2
            
        


    


filename = {'images' : '/home/abhinav/Desktop/Assignment 4_AT/train-images.idx3-ubyte' ,'labels' : '/home/abhinav/Desktop/Assignment 4_AT/train-labels.idx1-ubyte'}
#reading images data
train_imagesfile = open(filename['images'],'rb')
magicimages = st.unpack('>4B',train_imagesfile.read(4))
nImg = st.unpack('>I',train_imagesfile.read(4))[0] #num of images
nR = st.unpack('>I',train_imagesfile.read(4))[0] #num of rows
nC = st.unpack('>I',train_imagesfile.read(4))[0] #num of column
nBytesTotal = nImg*nR*nC*1 #since each pixel data is 1 byte
images = 255 - np.asarray(st.unpack('>'+'B'*nBytesTotal,train_imagesfile.read(nBytesTotal))).reshape((nImg,nR*nC))
#reading label data
train_labelfile = open(filename['labels'],'rb')
magiclabel = st.unpack('>4B',train_labelfile.read(4))
nLabels = st.unpack('>I',train_labelfile.read(4))[0] #num of labels
labels = np.asarray(st.unpack('>'+'B'*nLabels,train_labelfile.read(nLabels))).reshape((nLabels))
labelstemp=np.zeros((nLabels,10))
for i in range(nLabels):
    labelstemp[i][labels[i]]=1
labels=labelstemp

#training and validation split
train_x,dev_x,train_y,dev_y=train_test_split(images,labels,test_size=0.2,random_state=0)
#initial normalisation
train_x=batch_norm(train_x.T,0)
dev_x=batch_norm(dev_x.T,0)
train_y=train_y.T
dev_y=dev_y.T


#input parameter for a 2 layer ANN
nh=200 #number of hidden units in hidden layer
no=10 #number of units in output layer
b_size=500 #batch size for mini batch gradient descent
alpha=0.5 #learning rate
ITE=200

w1,w2,b1,b2=neural_net(train_x,train_y,dev_x,dev_y,nh,b_size,alpha,ITE)

#ACCURACY CALCULATION
a2=predict(w1,w2,b1,b2,train_x)
print("Training Accuracy:",round(accuracy(a2,train_y),2))
a2=predict(w1,w2,b1,b2,dev_x)
print("Validation Accuracy:",round(accuracy(a2,dev_y),2))

#test set accuracy
filename = {'images' : '/home/abhinav/Desktop/Assignment 4_AT/t10k-images.idx3-ubyte' ,'labels' : '/home/abhinav/Desktop/Assignment 4_AT/t10k-labels.idx1-ubyte'}
test_imagesfile = open(filename['images'],'rb')
magicimages = st.unpack('>4B',test_imagesfile.read(4))
nImg = st.unpack('>I',test_imagesfile.read(4))[0] #num of images
nR = st.unpack('>I',test_imagesfile.read(4))[0] #num of rows
nC = st.unpack('>I',test_imagesfile.read(4))[0] #num of column
nBytesTotal = nImg*nR*nC*1 #since each pixel data is 1 byte
images = 255 - np.asarray(st.unpack('>'+'B'*nBytesTotal,test_imagesfile.read(nBytesTotal))).reshape((nImg,nR*nC))
images=batch_norm(images.T,0)
test_labelfile = open(filename['labels'],'rb')
magiclabel = st.unpack('>4B',test_labelfile.read(4))
nLabels = st.unpack('>I',test_labelfile.read(4))[0] #num of labels
labels = np.asarray(st.unpack('>'+'B'*nLabels,test_labelfile.read(nLabels))).reshape((nLabels))
labelstemp=np.zeros((nLabels,10))
for i in range(nLabels):
    labelstemp[i][labels[i]]=1
labels=labelstemp.T
a2=predict(w1,w2,b1,b2,images)
print("Test Accuracy:",round(accuracy(a2,labels),2))










