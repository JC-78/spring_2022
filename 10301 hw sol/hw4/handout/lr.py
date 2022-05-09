import numpy as np
import sys
import time
import matplotlib.pyplot as plt

def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An N x 2 np.ndarray. N is the number of data points in the tsv file. The
        first column contains the label integer (0 or 1), and the second column
        contains the movie review string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8')
    #loop through dataset and get
    return dataset

def sigmoid(x):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)

def train(theta, X, y, num_epoch, learning_rate):
    # TODO: Implement `train` using vectorization
    
    for epoch in range(num_epoch):
        for feature,label in zip(X,y):
            a=np.multiply(learning_rate,feature)
            b=np.dot(theta,feature)
            c=label-sigmoid(b)
            theta=theta+np.multiply(a,c)
    return theta

def predict(theta, X):
    # TODO: Implement `predict` using vectorization
    length=len(X)
    res=np.zeros(length,dtype=float)
    count=0
    for feature in X:
        prob=sigmoid(np.dot(theta,feature))
        ans=0
        if prob>=0.5:
            ans=1
        res[count]=ans
        count+=1 #index update
    return res

def predict1(theta, X):
    # TODO: Implement `predict` using vectorization
    length=len(X)
    res=np.zeros(length,dtype=float)
    count=0
    probs=sigmoid(np.dot(theta,X))
    for prob in probs:
        ans=0
        if prob>=0.5:
            ans=1
        res[count]=ans
        count+=1 #index update
    return res

def compute_error(y_pred, y):
    # TODO: Implement `compute_error` using vectorization
    length=len(y_pred)
    if length!=len(y):
        print("length different")
        return False
    count=0 #count of mistakes
    for i in range(length):
        a=y_pred[i]
        b=y[i]
        if a!=b:
            count+=1
    return (count/length)

def load_data(data_path):
    with open(data_path, "r") as file:
        lines = file.readlines()
    length=len(lines)
    labels = np.zeros(length,dtype=int)
    features = np.zeros((length,14164+1)) #length of dict. ok to hardcode
    count=0
    for line in lines:
        instance = line.split("\t")
        labels[count]=float(instance[0])
        features[count][0]=1 #adding ones to feature vector to correspond to bias term.
        j=1
        for item in instance[1:]:
            features[count][j]=float(item) 
            j+=1
        count+=1
    
    return labels, features

def output(output_file, preds):
    item=''
    for pred in preds:
        item+=str(pred)+'\n'
    item=item.strip()
    with open(output_file, 'w') as res:
        res.write(item)

def loss_func(features,weights,labels):
    loss=0.0
    for feature,label in zip(features,labels):
        k=sigmoid(np.dot(weights,feature))
        #loss+=(label*k+sigmoid(k))
        loss+=(label*np.log(k))+(1-label)*np.log(1-k)
    return -loss/len(features)

def plot(train_features, train_labels, validation_features, validation_labels, num_epoch, learning_rate):
    theta=np.zeros(train_features.shape[1],dtype=np.float32)
    theta[0]=0
    train_loss=[]
    valid_loss=[]
    for epoch in range(num_epoch):
        for feature,label in zip(train_features,train_labels):
            a=np.multiply(learning_rate,feature)
            b=np.dot(theta,feature)
            c=label-sigmoid(b)
            theta=theta+np.multiply(a,c)
        loss1=loss_func(train_features,theta,train_labels)
        loss2=loss_func(validation_features,theta,validation_labels)
        train_loss.append(loss1)
        valid_loss.append(loss2)
    x = np.linspace(0, num_epoch - 1,num_epoch)
    plt.xlabel("Epochs")
    plt.ylabel("Negative Log Likelihood")
    plt.plot(x[:5000], train_loss[:5000], "r", linewidth = 3.0, label = "Training set")
    plt.plot(x[:5000], valid_loss[:5000], "b", linewidth = 3.0, label = "Validation set")
    plt.legend(loc='upper right')
    plt.show()
    return

def plot1(train_features, train_labels,num_epoch):
    theta=np.zeros(train_features.shape[1],dtype=np.float32)
    theta[0]=0
    train_loss=[]
    train_loss1=[]
    train_loss2=[]
    i=pow(10,-4)
    j=pow(10,-5)
    k=pow(10,-6)
    rates=[i,j,k]
    #Forgot to redefine theta. That's why issue happened.
    for rate in rates:
        for epoch in range(num_epoch):
            for feature,label in zip(train_features,train_labels):
                a=np.multiply(rate,feature)
                b=np.dot(theta,feature)
                c=label-sigmoid(b)
                theta=theta+np.multiply(a,c)
            loss1=loss_func(train_features,theta,train_labels)
            if rate==i:
                train_loss.append(loss1)
            if rate==j:
                train_loss1.append(loss1)
            if rate==k:
                train_loss2.append(loss1)
    x = np.linspace(0, num_epoch - 1,num_epoch)
    plt.xlabel("Epochs")
    plt.ylabel("Negative Log Likelihood")
    plt.plot(x[:5000], train_loss[:5000], "r", linewidth = 3.0, alpha=0.5,label = "Learning rate=10^-4")
    plt.plot(x[:5000], train_loss1[:5000], "b", linewidth = 3.0, alpha=0.5,label = "Learning rate=10^-5")
    plt.plot(x[:5000], train_loss2[:5000], "g", linewidth = 3.0, alpha=0.5,label = "Learning rate=10^-6")
    plt.legend(loc='upper right')
    plt.show()
    return

def plot2(train_features, train_labels, validation_features, validation_labels, num_epoch, learning_rate):
    theta=np.zeros(train_features.shape[1],dtype=np.float32)
    theta[0]=0
    valid_loss=[]
    for epoch in range(num_epoch):
        for feature,label in zip(train_features,train_labels):
            a=np.multiply(learning_rate,feature)
            #a=learning_rate*feature
            #np also has log. 
            b=np.dot(theta,feature)
            c=label-sigmoid(b)
            theta=theta+np.multiply(a,c)
        valid_loss.append(loss_func(validation_features,theta,validation_labels))
    x = np.linspace(0, num_epoch - 1,num_epoch)
    item=''
    for loss in valid_loss:
        item+=str(loss)+'\n'
    item=item.strip()
    with open("model 2 validation txt", 'w') as res:
        res.write(item)
    #plt.xlabel("Epochs")
    #plt.ylabel("Negative Log Likelihood")

    #plt.plot(x[:5000], valid_loss[:5000], "b", linewidth = 3.0, label = "Validation set")
    #plt.legend(loc='upper right')
    #plt.show()
    return

if __name__ == '__main__':
    #start = time.time() 
    formatted_train_input=str(sys.argv[1])
    formatted_validation_input=str(sys.argv[2])
    formatted_test_input=str(sys.argv[3])
    train_out=str(sys.argv[4])
    test_out=str(sys.argv[5])
    metrics_out=str(sys.argv[6])
    num_epoch=int(sys.argv[7])
    learning_rate=float(sys.argv[8])

    train_labels, train_features = load_data(formatted_train_input)
    #print(train_features[0])
    #data=load_tsv_dataset(formatted_train_input)
    #print(data[0])

    test_labels, test_features = load_data(formatted_test_input)
    validation_labels,validation_features = load_data(formatted_validation_input)
    """
    #print("dictionary is",train_features.shape[1]) 14165 since already folded in
    theta=np.zeros(train_features.shape[1],dtype=np.float32)
    theta[0]=0 #folding intercept parameter into parameter vector

    weights=train(theta, train_features, train_labels, num_epoch, learning_rate)
    pred=predict(weights,train_features)
    train_error=compute_error(pred,train_labels)
    pred1=predict(weights,test_features)
    test_error=compute_error(pred1,test_labels)
    
    file=open(metrics_out,"w")
    file.write(f"error(train): {train_error:.6f}\n")
    file.write(f"error(test): {test_error:.6f}\n")
    output(train_out,pred)
    output(test_out,pred1)
    """
    #plot2(train_features, train_labels, validation_features, validation_labels, num_epoch, learning_rate)
    #plot(train_features, train_labels, validation_features, validation_labels, num_epoch, learning_rate)
    plot1(train_features, train_labels,num_epoch)


    #end = time.time() 
    #print(end - start)



