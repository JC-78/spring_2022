import numpy as np
import argparse
import logging
import pdb
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


def args2data(parser):
    """
    Parse argument, create data and label.
    :return:
    X_tr: train data (numpy array)
    y_tr: train label (numpy array)
    X_te: test data (numpy array)
    y_te: test label (numpy array)
    out_tr: predicted output for train data (file)
    out_te: predicted output for test data (file)
    out_metrics: output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """

    # # Get data from arguments
    out_tr = parser.train_out
    out_te = parser.validation_out
    out_metrics = parser.metrics_out
    n_epochs = parser.num_epoch
    n_hid = parser.hidden_units
    init_flag = parser.init_flag
    lr = parser.learning_rate

    X_tr = np.loadtxt(parser.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr[:, 0] = 1.0 #add bias terms

    X_te = np.loadtxt(parser.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te[:, 0]= 1.0 #add bias terms


    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)

def shuffle(X, y, epoch):
    """
    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]

def random_init(shape):
    """
    Randomly initialize a numpy array of the specified shape
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    # DO NOT CHANGE THIS
    np.random.seed(np.prod(shape))

    # Implement random initialization here
    res=np.random.uniform(-0.1, 0.1, (shape[0], shape[1]))
    res[:,0]=0
    return res


def zero_init(shape):
    """
    Initialize a numpy array of the specified shape with zero
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    return np.zeros(shape)


class NN(object):
    def __init__(self, lr, n_epoch, weight_init_fn, input_size, hidden_size, output_size):
        """
        Initialization
        :param lr: learning rate
        :param n_epoch: number of training epochs
        :param weight_init_fn: weight initialization function
        :param input_size: number of units in the input layer
        :param hidden_size: number of units in the hidden layer
        :param output_size: number of units in the output layer
        """
        self.lr = float(lr)
        self.n_epoch = int(n_epoch)
        self.weight_init_fn = (zero_init if int(weight_init_fn)==2 else random_init)
        self.n_input = int(input_size)
        self.n_hidden = int(hidden_size)
        self.n_output = int(output_size)

        # initialize weights and biases for the models
        # HINT: pay attention to bias here.

        #of dimension D,M+1. D hidden units, let the input vectors x be of length M 4,130
        self.w1 = self.weight_init_fn([int(self.n_hidden), int(self.n_input)+1])
        #of dimension k, D+1. K classes, D+1 where D=# of hidden units
        self.w2 = self.weight_init_fn([10, self.n_hidden+1])
        #column alph=same, row beta=same.
        #bad to initialize as zero. 
        #same hidden unit=symmetry=

        # initialize parameters for adagrad
        self.epsilon = 1e-5 
        self.grad_sum_w1 =0
        self.grad_sum_w2 =0

        self.a=None
        self.z=None
        self.z1=None
        self.b=None

        
        # feel free to add additional attributes


def print_weights(nn):
    """
    An example of how to use logging to print out debugging infos.

    Note that we use the debug logging level -- if we use a higher logging
    level, we will log things with the default logging configuration,
    causing potential slowdowns.

    Note that we log NumPy matrices on separate lines -- if we do not do this,
    the arrays will be turned into strings even when our logging is set to
    ignore debug, causing potential massive slowdowns.
    :param nn: your model
    :return:
    """
    logging.debug(f"shape of w1: {nn.w1.shape}")
    logging.debug(nn.w1)
    logging.debug(f"shape of w2: {nn.w2.shape}")
    logging.debug(nn.w2)

def softmax(item):
    #print("item shape is ",item.shape)
    item=np.atleast_2d(item)
    #print("item shape is ",item.shape)
    item1=np.exp(item)
    #print("item1 shape is ",item.shape)
    out=item1/np.sum(item1,axis=1,keepdims=True)
    return out

def forward(x, nn):
    """
    Neural network forward computation.
    Follow the pseudocode!
    :param X: input data
    :param nn: neural network class
    :return: output probability
    """
    #print("x is",x.shape) #129,1= same as a row vector
    #print(nn.w1.shape) #4,129
    #print(nn.w2.shape) #10,5
    #a=np.dot(np.transpose(x),nn.w1)
    
    a=np.dot(x,np.transpose(nn.w1))
   
    #print("a is ",a.shape) #4,1
    e=np.exp(a)
    z=e/(1+e)
    #print("z's shape is",z.shape) #500,4
    
    bias=np.ones(1)

    #print("bias is ",bias.shape)
    #print(bias)
    #print(z)
    #print(type(bias),type(z))
    z1=np.concatenate((bias,z))
    #print("z1 is ", z1.shape)
    #z1=np.concatenate((bias,z),axis=1)
    #print("z1 shape after adding bias is",z1.shape)
    #print("nn.w2 is ",nn.w2.shape)
    b=np.dot(z1,np.transpose(nn.w2))
    #print(b.shape) #1,10
    #add softmax 
    res=softmax(b)
    #print("shape of res is", res.shape)
    #print(res)
    nn.a=a
    nn.z=z
    nn.z1=z1
    nn.b=b
    #nn.y_pred=res
    return res

def forward1(x, nn):
    """
    Neural network forward computation.
    Follow the pseudocode!
    :param X: input data
    :param nn: neural network class
    :return: output probability
    """
    #print("x is ",x.shape) #500,129
    #print(nn.w1.shape) #4,129
    a=np.dot(x,np.transpose(nn.w1))
    #print("a is ",a.shape)  100,4
    e=np.exp(a)
    z=e/(1+e)
    #print("z's shape is",z.shape) #500,4
    #print("x.shape[0] is", x.shape[0])
    
    bias=np.ones((x.shape[0],1))



    #print("bias shape is ",bias.shape)
    #print("bias is ",bias.shape)
    #print(bias)
    #print(z)
    #print(type(bias),type(z))
    z1=np.hstack((bias,z))
    
    #z1=np.concatenate((bias,z))
    #print("z1 is ", z1.shape)
    #print("nn.w2 is ",nn.w2.shape )
    b=np.dot(z1,np.transpose(nn.w2))
    #print("b is ", b.shape)
    res=softmax(b)

    #nn.a=a
    #nn.z=z1
    #nn.b=b
    return res

def backward(x, y, y_hat, nn):
    """
    Neural network backward computation.
    Follow the pseudocode!
    :param X: input data
    :param y: label
    :param y_hat: prediction
    :param nn: neural network class
    :return:
    d_w1: gradients for w1
    d_w2: gradients for w2
    """
    #pdb.set_trace()
    k=np.zeros((1,10))
    #k=np.zeros((1,3))
    index=y
    k[0][index]=1

    gb= y_hat-k
    
    #assert(gb.shape==(1,10))
    #print("Gb is", gb)
    
    #nn.z=nn.z.reshape((nn.n_hidden+1,1))
    nn.z1=nn.z1.reshape((nn.n_hidden+1,1))
    #print("gb is ",gb.shape)
    #print("nn.z is ",nn.z.shape)

    gbeta=np.dot(gb.T,nn.z1.T) #result still same after switching with matmul
    
    #assert(gbeta.shape==(10,5))
    #print("gbeta is ",gbeta.shape)
    #print("nn.w2 is ",nn.w2.shape)
    gz= np.dot(gb,nn.w2)
    gz=gz[:,1:] #disregarding bias 
    #assert(gz.shape==(1,4))
    
    #print("nn.z before chopping is",nn.z.shape)
    #nn.z=nn.z[1:,:] #4,1
    nn.z=nn.z.reshape((nn.n_hidden,1))
    #nn.z=nn.z[1:,:]
    #print("gz is",gz.shape)
    #print("nn.z is",nn.z.shape)
    ga=gz*nn.z.T*(1-nn.z.T)
    #print("ga is ",ga.shape)
    #0.0054=4e is
    #print("nn.z1 shape is ",nn.z1.shape)
    #print("nn.z shape is ",nn.z.shape)
    galpha=np.matmul(ga.T,x.reshape(1, -1))

    #print("galpha is ",galpha.shape)
    #never used nn.w1? use it to calculate gx but no need
    #print("galpha is ",galpha.shape)   same shape as alpha(1,4)
    #print("gbeta is ",gbeta.shape) same shape as beta (1,10).
    return galpha,gbeta

def test(X, y, nn,output,label_output,flag=0):
    """
    Compute the label and error rate.
    :param X: input data
    :param y: label
    :param nn: neural network class
    :return:
    labels: predicted labels
    error_rate: prediction error rate
    """
    length=X.shape[0]
    count=0
    yhat=forward1(X, nn)
    #print("yhat is ",yhat)    #0.01 for all indexes for all rows
    res=np.argmax(yhat,axis=1)
    #print("res is ",res)

    #output predicted labels using ame integer identifiers
    k=""
    for item in res:
        k+=str(item)+"\n"
    with open(label_output, 'w') as out:
        out.write(k)

    #print("res is ", res)      #due to above all 6
    a=np.array(y)
    same=np.equal(a,res)
    count=np.count_nonzero(same)
    error=(length-count)/length
    if flag==0:
        with open(output, 'a+') as out:
            out.seek(0)
            data=out.read(100)
            if len(data)>0:
                out.write("\n")
            res="error(train): "+str(error)
            out.write(res)
    if flag==1:
        with open(output, 'a+') as out:
            out.seek(0)
            data=out.read(100)
            if len(data)>0:
                out.write("\n")
                res="error(validation): "+str(error)
                out.write(res)
    return error

def train(X_tr, y_tr, X_te, y_te,nn,output):
    """
    Train the network using SGD for some epochs.
    :param X_tr: train data
    :param y_tr: train label
    :param nn: neural network class
    """
    nn.grad_sum_w1 =np.zeros([int(nn.n_hidden), int(nn.n_input)+1])
    nn.grad_sum_w2 =np.zeros([10, nn.n_hidden+1])
    
    #assert(nn.grad_sum_w1.shape==nn.w1.shape)
    #assert(nn.grad_sum_w2.shape==nn.w2.shape)
    #print("nn.grad_sum_w1 is",nn.grad_sum_w1.shape) 4,129
    #print("nn.grad_sum_w2 is",nn.grad_sum_w2.shape) 10,5
    
    res=""
    for e in range(nn.n_epoch):
        (X,Y)=shuffle(X_tr, y_tr, e)
        for (x,y) in zip(X,Y):
            
            yhat=forward(x,nn)
            d_w1,d_w2=backward(x,y,yhat,nn)
            #print("d_w1 is",d_w1)
            #print("d_w2 is",d_w2)
            nn.grad_sum_w1=nn.grad_sum_w1+(d_w1*d_w1)
            nn.grad_sum_w2=nn.grad_sum_w2+(d_w2*d_w2)
            nn.w1=nn.w1-(nn.lr/np.sqrt(nn.grad_sum_w1+nn.epsilon))*(d_w1)
            nn.w2=nn.w2-(nn.lr/np.sqrt(nn.grad_sum_w2+nn.epsilon))*(d_w2)
        yhat=forward1(X,nn)
        #print(Y.shape)
        #print("Y[1:5] is ",Y[1:5])
        #print("yhat[1:5 is ",yhat[1:5])
        ytrain_onehot=np.zeros(yhat.shape)
        ytrain_onehot[np.arange(len(Y)),Y]=1
        #print("ytrain_onehot is ", ytrain_onehot)
        #print("yhat is ",yhat.shape)
        CE= -np.sum(np.log(yhat)*ytrain_onehot)/len(Y)
        print("CE is ",CE)
        #CE=0
        res+="epoch="+str(e+1)+" crossentropy(train) : "+str(CE)+"\n"

        yhat1=forward1(X_te,nn)
        #print("yhat1[y_te] is ",yhat1[y_te][1:5])
        ytest_onehot=np.zeros(yhat1.shape)
        #print("ytest onehot is", ytest_onehot)
        ytest_onehot[np.arange(len(y_te)),y_te]=1

        CE1= -np.sum(np.log(yhat1)*ytest_onehot)/len(yhat1)
        print("CE1 is ",CE1)
        if e== (nn.n_epoch-1):
            res+="epoch="+str(e+1)+" crossentropy(validation) : "+str(CE1)
        else:
            res+="epoch="+str(e+1)+" crossentropy(validation) : "+str(CE1)+"\n" 
    with open(output, 'w') as out:
        out.write(res)

def empirical_train2a(X_tr, y_tr, X_te, y_te,nn):
    """
    Train the network using SGD for some epochs.
    :param X_tr: train data
    :param y_tr: train label
    :param nn: neural network class
    """
    nn.grad_sum_w1 =np.zeros([int(nn.n_hidden), int(nn.n_input)+1])
    nn.grad_sum_w2 =np.zeros([10, nn.n_hidden+1])
    
    #assert(nn.grad_sum_w1.shape==nn.w1.shape)
    #assert(nn.grad_sum_w2.shape==nn.w2.shape)
    #print("nn.grad_sum_w1 is",nn.grad_sum_w1.shape) 4,129
    #print("nn.grad_sum_w2 is",nn.grad_sum_w2.shape) 10,5
    
    res=[]
    res1=[]
    ans=0
    ans1=0
    for e in range(nn.n_epoch):
        (X,Y)=shuffle(X_tr, y_tr, e)
        for (x,y) in zip(X,Y):
            
            yhat=forward(x,nn)
            d_w1,d_w2=backward(x,y,yhat,nn)
            #print("d_w1 is",d_w1)
            #print("d_w2 is",d_w2)
            nn.grad_sum_w1=nn.grad_sum_w1+(d_w1*d_w1)
            nn.grad_sum_w2=nn.grad_sum_w2+(d_w2*d_w2)
            nn.w1=nn.w1-(nn.lr/np.sqrt(nn.grad_sum_w1+nn.epsilon))*(d_w1)
            nn.w2=nn.w2-(nn.lr/np.sqrt(nn.grad_sum_w2+nn.epsilon))*(d_w2)
        yhat=forward1(X,nn)
        #print(Y.shape)
        #print("Y[1:5] is ",Y[1:5])
        #print("yhat[1:5 is ",yhat[1:5])
        ytrain_onehot=np.zeros(yhat.shape)
        ytrain_onehot[np.arange(len(Y)),Y]=1
        #print("ytrain_onehot is ", ytrain_onehot)
        #print("yhat is ",yhat.shape)
        CE= -np.sum(np.log(yhat)*ytrain_onehot)/len(Y)
        ans=CE
        average_train_CE=ans/len(X_tr)
        #res.append(ans/(e+1))
        res.append(average_train_CE)

        yhat1=forward1(X_te,nn)
        ytest_onehot=np.zeros(yhat1.shape)
        ytest_onehot[np.arange(len(y_te)),y_te]=1
        CE1= -np.sum(np.log(yhat1)*ytest_onehot)/len(yhat1)
        ans1=CE1
        average_test_CE=ans1/len(X_te)
        res1.append(average_test_CE)
    return res,res1

def empirical_train(X_tr, y_tr, X_te, y_te,nn):
    """
    Train the network using SGD for some epochs.
    :param X_tr: train data
    :param y_tr: train label
    :param nn: neural network class
    """
    nn.grad_sum_w1 =np.zeros([int(nn.n_hidden), int(nn.n_input)+1])
    nn.grad_sum_w2 =np.zeros([10, nn.n_hidden+1])
    
    #assert(nn.grad_sum_w1.shape==nn.w1.shape)
    #assert(nn.grad_sum_w2.shape==nn.w2.shape)
    #print("nn.grad_sum_w1 is",nn.grad_sum_w1.shape) 4,129
    #print("nn.grad_sum_w2 is",nn.grad_sum_w2.shape) 10,5
    
    res=""
    #ans=0
    #ans1=0
    for e in range(nn.n_epoch):
        (X,Y)=shuffle(X_tr, y_tr, e)
        for (x,y) in zip(X,Y):
            
            yhat=forward(x,nn)
            d_w1,d_w2=backward(x,y,yhat,nn)
            #print("d_w1 is",d_w1)
            #print("d_w2 is",d_w2)
            nn.grad_sum_w1=nn.grad_sum_w1+(d_w1*d_w1)
            nn.grad_sum_w2=nn.grad_sum_w2+(d_w2*d_w2)
            nn.w1=nn.w1-(nn.lr/np.sqrt(nn.grad_sum_w1+nn.epsilon))*(d_w1)
            nn.w2=nn.w2-(nn.lr/np.sqrt(nn.grad_sum_w2+nn.epsilon))*(d_w2)
        yhat=forward1(X,nn)
        #print(Y.shape)
        #print("Y[1:5] is ",Y[1:5])
        #print("yhat[1:5 is ",yhat[1:5])
        ytrain_onehot=np.zeros(yhat.shape)
        ytrain_onehot[np.arange(len(Y)),Y]=1
        #print("ytrain_onehot is ", ytrain_onehot)
        #print("yhat is ",yhat.shape)
        #CE= -np.sum(np.log(yhat)*ytrain_onehot)/len(Y)
        CE= -np.sum(np.log(yhat)*ytrain_onehot)
        #ans+=CE
        yhat1=forward1(X_te,nn)
        ytest_onehot=np.zeros(yhat1.shape)
        ytest_onehot[np.arange(len(y_te)),y_te]=1

        #CE1= -np.sum(np.log(yhat1)*ytest_onehot)/len(yhat1)
        CE1= -np.sum(np.log(yhat1)*ytest_onehot)
        #ans1+=CE
    average_train_CE=CE/len(X_tr)
    average_test_CE=CE1/len(X_te)
    #average_train_CE=ans/len(X_tr)
    #average_test_CE=ans/len(X_te)
    return average_train_CE,average_test_CE
        
def empirical_1c(X_tr, y_tr, X_te, y_te,nn):
    """
    Train the network using SGD for some epochs.
    :param X_tr: train data
    :param y_tr: train label
    :param nn: neural network class
    """
    nn.grad_sum_w1 =np.zeros([int(nn.n_hidden), int(nn.n_input)+1])
    nn.grad_sum_w2 =np.zeros([10, nn.n_hidden+1])
    
    #assert(nn.grad_sum_w1.shape==nn.w1.shape)
    #assert(nn.grad_sum_w2.shape==nn.w2.shape)
    #print("nn.grad_sum_w1 is",nn.grad_sum_w1.shape) 4,129
    #print("nn.grad_sum_w2 is",nn.grad_sum_w2.shape) 10,5
    
    res=""
    k=[]
    for e in range(nn.n_epoch):
        (X,Y)=shuffle(X_tr, y_tr, e)
        for (x,y) in zip(X,Y):
            
            yhat=forward(x,nn)
            d_w1,d_w2=backward(x,y,yhat,nn)
            #print("d_w1 is",d_w1)
            #print("d_w2 is",d_w2)
            nn.grad_sum_w1=nn.grad_sum_w1+(d_w1*d_w1)
            nn.grad_sum_w2=nn.grad_sum_w2+(d_w2*d_w2)
            nn.w1=nn.w1-(nn.lr/np.sqrt(nn.grad_sum_w1+nn.epsilon))*(d_w1)
            nn.w2=nn.w2-(nn.lr/np.sqrt(nn.grad_sum_w2+nn.epsilon))*(d_w2)
        yhat=forward1(X,nn)
        ytrain_onehot=np.zeros(yhat.shape)
        ytrain_onehot[np.arange(len(Y)),Y]=1
        CE= -np.sum(np.log(yhat)*ytrain_onehot)/len(Y)
        yhat1=forward1(X_te,nn)
        ytest_onehot=np.zeros(yhat1.shape)
        ytest_onehot[np.arange(len(y_te)),y_te]=1
        CE1= -np.sum(np.log(yhat1)*ytest_onehot)/len(yhat1)
        k.append(CE1)
    return k

if __name__ == "__main__":

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    # Note: You can access arguments like learning rate with args.learning_rate
    
    # initialize training / test data and labels
    X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,n_epochs, n_hid, init_flag, lr=args2data(parser.parse_args())
    #print(lr)
    
    #hw code
    #my_nn = NN(lr,n_epochs,init_flag,X_tr.shape[1]-1,n_hid,1)
    #train(X_tr, y_tr,X_te, y_te,my_nn,args.metrics_out)
    #test(X_tr, y_tr, my_nn,args.metrics_out,args.train_out,0)
    #test(X_te, y_te, my_nn,args.metrics_out,args.validation_out,1)

    #code for empirical
    #rates=[0.1,0.01,0.001]
    my_nn = NN(lr,n_epochs,init_flag,X_tr.shape[1]-1,50,1)
    res=[]
    with open('metrics_sgd_small.txt') as f:
        lines = f.readlines()
        lines=lines[:-2]
        count=0
        for item in lines:
            if (count%2)!=0:
                item=item.split(" ")
                a=item[-1].replace('\n',"")
                #print(a)
                res.append(np.float64(a))
            count+=1

        #print(res)
        #print(res[-1])
    k=empirical_1c(X_tr, y_tr, X_te, y_te,my_nn)
    x = np.linspace(0, n_epochs - 1,n_epochs)
    #print(x)
    #print("res is ", res)
    #print("k is ",k)
    plt.xlabel("# of epoch")
    plt.ylabel("Cross Entropy")
    plt.plot(x[:100], res[:100], "r", linewidth = 3.0, alpha=0.5,label = "SGD. No Adagrad")
    plt.plot(x[:100], k[:100], "b", linewidth = 3.0, alpha=0.5,label = "Adagrad")
    plt.title("The relationship between validation results for SGD \n with and without Adagrad")
    plt.legend(loc='upper right')
    plt.show()

    """
    rates=[0.001]
    lr1_train=[]
    lr1_test=[]
    lr2_train=[]
    lr2_test=[]
    lr3_train=[]
    lr3_test=[]
    for rate in rates:
        my_nn = NN(rate,n_epochs,init_flag,X_tr.shape[1]-1,50,1)
        train_entropy,validation_entropy=empirical_train2a(X_tr, y_tr, X_te, y_te,my_nn)
        if rate==0.1:
            lr1_train=train_entropy
            #print("lr1_train is",lr1_train) 
            lr1_test=validation_entropy
            #print("lr1_test is",lr1_test)
        if rate==0.01:
            lr2_train=train_entropy
            lr2_test=validation_entropy
        if rate==0.001:
            lr3_train=train_entropy
            lr3_test=validation_entropy
    x = np.linspace(0, n_epochs - 1,n_epochs)
    #print("alpha is ", my_nn.w1)
    #print("beta is ", my_nn.w2)
    #print(x)
    #print("res is ", res)
    #print("k is ",k)
    plt.xlabel("# of epoch")
    plt.ylabel("Cross Entropy")
    #print(len(x))
    #print(len(lr1_train))
    #print(len(lr1_test))
    #plt.plot(x[:100], lr1_train[:100], "b", linewidth = 3.0, alpha=0.5,label = "lr=0.1 train")
    #plt.plot(x[:100], lr1_test[:100], "g", linewidth = 3.0, alpha=0.5,label = "lr=0.1 test")
    #plt.plot(x[:100], lr2_train[:100], "r", linewidth = 3.0, alpha=0.5,label = "lr=0.01 train")
    #plt.plot(x[:100], lr2_test[:100], "c", linewidth = 3.0, alpha=0.5,label = "lr=0.01 test")
    plt.plot(x[:100], lr3_train[:100], "m", linewidth = 3.0, alpha=0.5,label = "lr=0.001 train")
    plt.plot(x[:100], lr3_test[:100], "y", linewidth = 3.0, alpha=0.5,label = "lr=0.001 test")
    plt.title("Average training and validation cross-entropy \n loss for lr=0.001")
    plt.legend(loc='upper right')
    plt.show()
    """




    """1c
    my_nn = NN(lr,n_epochs,init_flag,X_tr.shape[1]-1,50,1)
    res=[]
    with open('metrics_sgd_small.txt') as f:
        lines = f.readlines()
        lines=lines[:-2]
        count=0
        for item in lines:
            if (count%2)!=1:
                item=item.split(" ")
                a=item[-1].replace('\n',"")
                #print(a)
                res.append(np.float64(a))
            count+=1

        #print(res)
        #print(res[-1])
    k=empirical_1c(X_tr, y_tr, X_te, y_te,my_nn)
    x = np.linspace(0, n_epochs - 1,n_epochs)
    #print(x)
    #print("res is ", res)
    #print("k is ",k)
    plt.xlabel("# of epoch")
    plt.ylabel("Cross Entropy")
    plt.plot(x[:100], res[:100], "r", linewidth = 3.0, alpha=0.5,label = "without Adagrad")
    plt.plot(x[:100], k[:100], "b", linewidth = 3.0, alpha=0.5,label = "Adagrad")
    plt.title("The relationship between validation results for SGD \n with and without Adagrad")
    plt.legend(loc='upper right')
    plt.show()
    """


    """1a
    hidden=[5,20,50,100,200]
    train_loss=[]
    test_loss=[]
    for h in hidden:
        my_nn = NN(lr,n_epochs,init_flag,X_tr.shape[1]-1,h,1)
        average_train_CE,average_test_CE=empirical_train(X_tr, y_tr,X_te, y_te,my_nn)
        train_loss.append(average_train_CE)
        test_loss.append(average_test_CE)
    plt.xlabel("Hidden Units")
    plt.ylabel("Average Cross Entropy")
    plt.plot(hidden[:5], train_loss[:5], "r", linewidth = 3.0, alpha=0.5,label = "Train loss")
    plt.plot(hidden[:5], test_loss[:5], "b", linewidth = 3.0, alpha=0.5,label = "Test loss")
    plt.title("The relationship between hidden units and average cross entropy")
    plt.legend(loc='upper right')
    plt.show()
    """

    
