
import sys
import numpy as np
import math


def read_data(input_file):
    """
    reads data and returns data's labels. 
    """
    with open(input_file, 'r') as data:
        data=[line.split() for line in data]    
    data=data[1:]
    labelIndex=len(data[0])-1
    labels=[row[labelIndex] for row in data]
    #print(labels)
    #print(len(labels))
    return labels

def get_metrics(labels):
    d=dict()
    for item in labels:
        if item not in d:
            d[item]=1
        else:
            d[item]+=1
    entropy=0
    maxItem=""
    maxCount=-1
    length=len(labels)
    for item in d:
        count=d[item]
        prob=count/length
        value= (-1*prob)*math.log(prob,2)
        entropy+=value
        if count>maxCount:
            maxItem=item
            maxCount=count
    
    error=(length-maxCount)/length
    return entropy,error

def output(output_file, entropy, error):
    """
    Produces the output file with entropy and error written
    """
    ans='entropy: '+str(entropy)+"\n"+"error: "+str(error)
    with open(output_file, 'w') as res:
        res.write(ans)

#conda activate myenvname
if __name__ == '__main__':
    input_file=sys.argv[1]
    labels=read_data(input_file)
    output_file=sys.argv[2]
    entropy,error=get_metrics(labels)
    output(output_file,entropy,error)