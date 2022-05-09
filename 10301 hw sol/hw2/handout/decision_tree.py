import sys
import numpy as np
import math
import copy

class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self,data,attr=None,depth=0,used=set()):
        self.left = None
        self.right = None
        self.vote = None
        self.data=data
        self.attr=attr
        self.depth=depth
        self.used=used
        self.child=dict()
        
def read_data(input_file):
    """
    reads data and returns 
    1)data in dictionary format 
    2)attributes and their order
    """
    with open(input_file, 'r') as data:
        data=[line.split() for line in data]  
    k=data[0]  #first row
    column_count=len(k)
    features=dict()
    features[-1]=k[column_count-1] #labels
    data1=dict()
    for feature in k:
        data1[feature] = []
    for r in data[1:]:
        for c in range(column_count):
            features[k[c]]=c #attribute column order number
            data1[k[c]].append(r[c])
    #print(data1)
    #print(attributes)
    return (data1, features)

def majority(data,label):
    """
    gets the MajorityVoteItem in the dataset 
    """
    d=dict()
    for item in data[label]:
        if item not in d:
            d[item]=1
        else:
            d[item]+=1
    maxItem="NA"
    maxCount=-1
    for item in d:
        if d[item]>maxCount:
            maxItem=item
            maxCount=d[item]
        elif d[item]==maxCount:
            a=maxItem[0] #democrat
            b=item[0] #republic
            if a[0]<b[0]:
                maxItem=item
    return maxItem,maxCount

def entropy_help(labels,feature=None):
    length=len(labels)
    label_values=set(labels)
    if length==0 or len(label_values)==1:
        return 0
    if feature!=None:
        #conditional entropy
        label_values=set(feature)
        label_tmp=labels
        labels=feature
        feature=label_tmp
    d=dict()
    for item in label_values:
        d[item]=[]
    count=0
    for i in range(length):
        label=labels[i]
        if feature:
            d[label].append(feature[i])
        else:
            d[label].append(label)
        count+=1
    res=0
    for label in d:
        p=len(d[label])/count
        entropy=0
        if p!=0:
            value=math.log(p, 2)
            entropy=p*value*-1
        if feature:
            entropy1=entropy_help(d[label])
            entropy=p*entropy1
        res+=entropy
    return res

def MI_help(data, label, feature):
    entropy1=entropy_help(data[label], data[feature])
    entropy2=entropy_help(data[label])
    res=entropy2-entropy1
    return res

def createDT(data,data_attributes,max_depth,depth,used=set()):
    end=0
    label=data_attributes[-1]
    length=len(data)-1 
    #base case where max_depth=0
    if (max_depth==end) or (len(used)==length) or (max_depth==depth):
        ans=Node(copy.deepcopy(data),label,depth, used)
        ans.vote,vote_count=majority(data,label)
        return ans 

    best_split=None
    MI_max=-1
    for feature in data:
        if feature!=label:
            MI=MI_help(data, label, feature)
        if MI>0 and MI_max==MI:
            #If tied, then split on the first column to break ties
            a=data_attributes[feature]
            b=data_attributes[best_split]
            if a<b: 
                best_split=feature
        elif MI>0 and MI>MI_max:
            MI_max=MI
            best_split=feature 
       
    if not best_split:
        leaf_node=Node(copy.deepcopy(data),label,depth, used)
        leaf_node.vote,vote_count1=majority(data,label)
        return leaf_node
    
    partitioned_data=dict()
    k=data[best_split]
    length=len(k)
    attribute_values=set(k)
    for value in attribute_values:
        partitioned_data[value]=dict()
        for feature in data:
            partitioned_data[value][feature]=[]
    for i in range(length):
        item=k[i]
        for feature in data:
            partitioned_data[item][feature].append(data[feature][i])
    
    new_used=used.copy()
    new_used.add(best_split)
    new_depth=depth+1
    res_tree=Node(copy.deepcopy(data),best_split,depth, new_used)
    
    for item in partitioned_data:
        data1 = copy.deepcopy(partitioned_data[item])
        res_tree.child[item]=createDT(data1, data_attributes, max_depth, new_depth, new_used)
    return res_tree

def printDT(tree, label_name, labels, depth = 0):
    s='['
    maxItem, maxCount=majority(tree.data,label_name)
    count=len(tree.data[label_name])
    for label in labels:
        if label!=maxItem:
            val=count-maxCount
            s+=str(val)
            s+=' '+label+' / '
        else:
            s+=str(maxCount)
            s+=' '+label+' / '            
    s=s[:-2]+'] '
    print(s)

    if type(tree.vote)!= None:
        new_depth=depth+1
        tmp=(' | ' * new_depth)
        tmp=tmp+tree.attr+' = '
        items=tree.child.keys()
        values = list(items)
        for value in values:
            s=tmp+value+' : '
            print(s, end = " ")
            new=tree.child[value]
            printDT(new, label_name, labels, new_depth)

def predict(model, data):
    check=False
    if type(model.vote)==str:
        check=True
    if check:
        return model.vote
    new_model=model.child[data[model.attr]]
    return predict(new_model,data)

def test(DT,data,labels):
    length=len(labels)
    error=0
    pred=[]
    #pred=np.array([])
    for i in range(length):
        d=dict()
        for feature in data:
            k=data[feature][i]
            d[feature]=k
        guess=predict(DT,d)
        pred.append(guess)
        #np.append(pred,guess)
        #print("meow",i)
        #print(pred)
        if guess!=labels[i]:
            error+=1
    #pred=list(pred)
    #print(pred)
    return (pred,error/length)

def metric_output(name,train_error,test_error):
    res="error(train): "+str(train_error)+"\n"
    res+="error(test): "+str(test_error)+"\n"
    with open(name, 'w') as out:
        out.write(res)

def output(output_file, preds):
    item=''
    for pred in preds:
        item+=pred+'\n'
    item=item.strip()
    with open(output_file, 'w') as res:
        res.write(item)

if __name__ == '__main__':
    train_input=sys.argv[1]
    train_data,train_attributes=read_data(train_input)

    max_depth=int(sys.argv[3])
    classifier=createDT(train_data,train_attributes,max_depth,depth=0)

    test_input=sys.argv[2]
    test_data,test_attributes=read_data(test_input)
    label_column_index=-1
    label_name=train_attributes[label_column_index]
    train_labels=train_data[label_name]
    s=set(train_labels)
    labels=list(s)
    labels.sort()

    printDT(classifier, label_name,labels)

    test_labels=test_data[test_attributes[label_column_index]]
    train_pred,train_error=test(classifier, train_data, train_labels)
    test_pred,test_error=test(classifier, test_data, test_labels)

    metrics_out=sys.argv[6]
    metric_output(metrics_out,train_error,test_error)
    train_out=sys.argv[4]
    test_out=sys.argv[5]

    output(train_out,train_pred)
    output(test_out,test_pred)
