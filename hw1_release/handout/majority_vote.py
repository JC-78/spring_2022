# -*- coding: utf-8 -*- 
import sys
import numpy as np

def read_data(input_file):
    """
    reads data and returns data without column names
    """
    data = np.genfromtxt(input_file, delimiter="\t", dtype=None)
    title = data[0, 0:data.shape[1]]
    data = data[1:]
    return data

def train_model(data,value):
    """
    gets the MajorityVoteItem and its count in the training set 
    """
    d=dict()
    for i in range(value):
        label=data[i,-1]
        if label not in d:
            d[label]=1
        else:
            d[label]+=1
    maxItem="NA"
    maxCount=-1
    for item in d:
        if d[item]>maxCount:
            maxItem=item
            maxCount=d[item]
        elif d[item]==maxCount:
            a=maxItem[0]
            b=item[0]
            if a<b:
                maxItem=item
    return maxItem,maxCount

def predict(MajorityLabel,value):
    """
    returns the numpy array of MajorityLabel, which is as long as the data
    """
    res=np.array([MajorityLabel for _ in range(value)])
    for i in range(value):
        res[i]=MajorityLabel
    return res
    
def output(name, pred):
    """
    creates an output file
    """
    res = ''
    for label in pred:
        res+=label.decode('UTF-8')+"\n"
    with open(name, 'w') as out:
        out.write(res)

def testError(test_data,rows,MajorityVoteItem):
    count=0
    for i in range(rows):
        if test_data[i,-1]!=MajorityVoteItem:
            count+=1
    outcome= count/rows
    return outcome

def metric_output(name,train_error,test_error):
    res="error(train): "+str(train_error)+"\n"
    res+="error(test): "+str(test_error)+"\n"
    with open(name, 'w') as out:
        out.write(res)

"""
python majority_vote.py politicians_train.tsv politicians_test.tsv \
pol_train.labels pol_test.labels pol_metrics.txt

myenvname
"""
if __name__ == "__main__":
    train = sys.argv[1]
    test = sys.argv[2]
    
    train1=read_data(train)
    rows=train1.shape[0]
    MajorityVoteItem,MajorityCount=train_model(train1,rows)
    prediction=predict(MajorityVoteItem,rows)
    name=sys.argv[3] #ex.pol_train
    output(name,prediction)
    train_error= (rows-MajorityCount)/rows
    print("train_error is ",train_error)

    test1=read_data(test)
    rows1=test1.shape[0]
    name1=sys.argv[4] #ex.pol_test
    prediction1=predict(MajorityVoteItem,rows1)
    output(name1,prediction1)
    test_error=testError(test1,rows1,MajorityVoteItem)
    print("test_error is" ,test_error)

    name2=sys.argv[5]
    metric_output(name2,train_error,test_error)


"""
Remaining things to handle:

-The training procedure should store the label used for prediction at test time. 
In the case of a tie, output the value that comes last alphabetically.
#Does this mean if tied between two options, choose the one that starts off with later alphabet? 
ex.alpha vs bravo? use bravo 

-submit majority_vote.py
Infinite submission. Only last one will be graded. 

Currently only failing dataset two out of four so that's good. 
"""