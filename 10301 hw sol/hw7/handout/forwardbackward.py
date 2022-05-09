############ Welcome to HW7 ############
# TODO: Andrew-id: 


# Imports
# Don't import any other library

import numpy as np
import sys
from utils import make_dict, parse_file, get_matrices, write_predictions, write_metrics
import argparse
import logging

# Setting up the argument parser
# don't change anything here
parser = argparse.ArgumentParser()
parser.add_argument('validation_input', type=str,
                    help='path to validation input .txt file')
parser.add_argument('index_to_word', type=str,
                    help='path to index_to_word.txt file')
parser.add_argument('index_to_tag', type=str,
                    help='path to index_to_tag.txt file')
parser.add_argument('init', type=str,
                    help='path to the learned hmm_init.txt (pi) file')
parser.add_argument('emission', type=str,
                    help='path to the learned hmm_emission.txt (A) file')
parser.add_argument('transition', type=str,
                    help='path to the learned hmm_transition.txt (B) file')
parser.add_argument('prediction_file', type=str,
                    help='path to store predictions')
parser.add_argument('metric_file', type=str,
                    help='path to store metrics')                    
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')



# Hint: You might find it helpful to define functions 
# that do the following:
# 1. Calculate Alphas
# 2. Calculate Betas
# 3. Implement the LogSumExpTrick
# 4. Calculate probabilities and predictions

DEBUG = False

def dbg_print(*args):
    if DEBUG:
        print(args)

def logSumExpTrick(v):
    m = np.max(v)
    return m + np.log(sum(np.exp(v - m)))

def helper(words,word_dict,tag_dict,init,emission,transition):
    words_length=len(words)
    tag_dict_length=len(tag_dict)
    a=np.zeros((words_length,tag_dict_length))
    b=np.zeros((words_length,tag_dict_length))
    
    k=word_dict[words[0]]
    for i in range(tag_dict_length):
        #print("i is ",i)
        #print("k is ",k)
        #print("emission shape is ",emission.shape)
        k1=emission[i][k]
        value=init[i]*k1
        #a[0][i]=value
        #a[0][i]=np.log(value)
        a[0][i]=np.log(init[i])+np.log(k1)
    
    end=words_length-1
    for t in range(1,words_length):
        for j in range(tag_dict_length):
            #res=0 correct
            res=[]
            for i in range(tag_dict_length):
                value=a[t-1][i]
                value1=transition[i][j]
                #res+=(value*value1) #correct
                res.append(value+np.log(value1))
                
            index=word_dict[words[t]]
            #a[t][j]=emission[j][index]*res 
            a[t][j]=np.log(emission[j][index])+logSumExpTrick(res)
    
    log_likelihood=logSumExpTrick(a[-1])

    # for t in range(words_length):
    #     a[t]/=np.sum(a[t])

    for j in range(tag_dict_length):
        # b[-1][j]=1
        b[-1][j]=0
    #print(b)
    for t in range(words_length-2,-1,-1):
        for j in range(0,tag_dict_length):
            res2=[]
            for index in range(0,tag_dict_length):
                item_index=word_dict[words[t+1]]
                #b[t][j]+=emission[index][item_index]*b[t+1][index]*transition[j][index]  correct
                #b[t][j]+=np.log(emission[index][item_index]*b[t+1][index]*transition[j][index]) 
                #b[t][j]+=np.log(emission[index][item_index])+np.log(b[t+1][index]*transition[j][index])
                res2.append( np.log(emission[index][item_index]) + (b[t+1][index]) + (np.log(transition[j][index])) )
            b[t][j]=logSumExpTrick(res2)
    k=b.shape[0]

    #print("b is ",b)
    #print("np.sum(b,axis=1).reshape(k,-1) is ",np.sum(b,axis=1).reshape(k,-1))
    #print("division result is ", b/np.sum(b,axis=1).reshape(k,-1))
    
    # b/=np.sum(b,axis=1).reshape(k,-1)
    
    #res=a*b
    #print("a is ",a)
    #print("b is ",b)
    res=a+b
    #print("res is ",res)
    ans_index=np.argmax(res,axis=1)
    #print("res", res)
    #print("ans index", ans_index)
    reverse_index = {v:k for (k,v) in tag_dict.items()}
    final=[]
    count=0
    for item in ans_index:
        # for item1 in tag_dict:
        #     print("item is ",item)
        #     print("item1 is ",item1)
        #     if tag_dict[item1]==ans_index[count]:
        #         final.append(item1)
        #         count+=1
        #         break
        final.append(reverse_index[item])
    return final,log_likelihood


def analysis(sentences,tags,word_dict,tag_dict,init,emission,transition):
    ans=0
    pred=[]
    correct=0
    tags_count=0
    count=0
    #print("tags are ",tags)
    #print("sentences are ",sentences)
    for sent in sentences:
        k=tags[count]
        length=len(k)
        tags1,log_likelihood=helper(sent,word_dict,tag_dict,init,emission,transition)
        pred.append(tags1)
        ans+=log_likelihood
        tags_count+=len(tags1)
        dbg_print("tags1 length is ",len(tags1))
        dbg_print("k length is ",len(k))
        for i in range(length):
            #print("tags are ", k)
            #print("tags1 are ", tags1)
            #print("tags shape is ",len(k))
            #print("tags1 shape is ",len(tags1))
            #print("count is ",count," and i is ",i )
            #print("i is ",i)
            if k[i]==tags1[i]:
                correct+=1
        count+=1
    avg_log_likelihood=ans/len(sentences)
    accuracy=correct/tags_count
    print(avg_log_likelihood,accuracy)
    return pred,avg_log_likelihood,accuracy
        



# TODO: Complete the main function
def main(args):

    # Get the dictionaries
    word_dict = make_dict(args.index_to_word)
    tag_dict = make_dict(args.index_to_tag)

    # Parse the validation file
    sentences, tags = parse_file(args.validation_input)
    
    #empirical question part 
    k=len(sentences)
    sentences=sentences[0:10]
    tags=tags[0:10]




    # Load your learned matrices
    # Make sure you have them in the right orientation
    init, emission, transition = get_matrices(args)
    #print("init is ", init)

    
    # TODO: Conduct your inferences
    
    # TODO: Generate your probabilities and predictions
    a,b,c=analysis(sentences,tags,word_dict,tag_dict,init,emission,transition)

  
    predicted_tags = a#TODO: store your predicted tags here (in the right order)
    avg_log_likelihood = b# TODO: store your calculated average log-likelihood here
    
    accuracy = c # We'll calculate this for you

    # Writing results to the corresponding files.  
    # We're doing this for you :)
    accuracy = write_predictions(args.prediction_file, sentences, predicted_tags, tags)
    write_metrics(args.metric_file, avg_log_likelihood, accuracy)

    return

if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    main(args)
