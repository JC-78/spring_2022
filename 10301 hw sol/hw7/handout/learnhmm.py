############ Welcome to HW7 ############
# TODO: Andrew-id: 


# Imports
# Don't import any other library

import argparse
import numpy as np
import sys
from utils import make_dict, parse_file
import logging

# Setting up the argument parser
# don't change anything here

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .txt file')
parser.add_argument('index_to_word', type=str,
                    help='path to index_to_word.txt file')
parser.add_argument('index_to_tag', type=str,
                    help='path to index_to_tag.txt file')
parser.add_argument('init', type=str,
                    help='path to store the hmm_init.txt (pi) file')
parser.add_argument('emission', type=str,
                    help='path to store the hmm_emission.txt (A) file')
parser.add_argument('transition', type=str,
                    help='path to store the hmm_transition.txt (B) file')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


# Hint: You might find it useful to define functions that do the following:
# 1. Calculate the init matrix
# 2. Calculate the emission matrix
# 3. Calculate the transition matrix
# 4. Normalize the matrices appropriately

def construct(sentences,tags,word_dict,tag_dict):
    #print("sentences are ", sentences)
    #print("tags are ", tags)
    #print("word dict is ", word_dict)
    #print("tag_dict is ",tag_dict)
    
    length=len(tag_dict)
    init=np.zeros(length)
    transition=np.zeros((length,length))
    length1=len(word_dict)
    emission=np.zeros((length,length1))
    #print("init.shape is ",init.shape)
    #print("transition.shape is ",transition.shape)
    #print("emission.shape is ",emission.shape)
    count=0
    for sent in sentences:
        length2=len(sent)
        for i,item in enumerate(sent):
            word_index=word_dict[item]
            tag=tags[count][i]
            tag_index=tag_dict[tag]
            if i==0:
                init[tag_index]+=1
            if i!=length2-1:
                k=tags[count][i+1]
                next_tag_index=tag_dict[k]
                transition[tag_index][next_tag_index]+=1
            emission[tag_index][word_index]+=1
        count+=1
    pseudo=1
    init+=pseudo
    init/=np.sum(init)
    transition+=pseudo
    #print("transition shape is ",transition.shape)
    #print("length shape is ",length)
    transition/=np.sum(transition,axis=1).reshape(length,-1)
    emission+=pseudo
    emission/=np.sum(emission,axis=1).reshape(length,-1)
    #print(init,transition,emission)
    #print("init.shape1 is ",init.shape)
    #print("transition.shape1 is ",transition.shape)
    #print("emission.shape1 is ",emission.shape)
    return init,emission,transition
# TODO: Complete the main function
def main(args):

    # Get the dictionaries
    word_dict = make_dict(args.index_to_word)
    tag_dict = make_dict(args.index_to_tag)

    # Parse the train file
    # Suggestion: Take a minute to look at the training file,
    # it always hels to know your data :)
    sentences, tags = parse_file(args.train_input) #changed from train_file to train_input
    
    #empirical portion
    sentences=sentences[0:10]
    tags=tags[0:10]

    logging.debug(f"Num Sentences: {len(sentences)}")
    logging.debug(f"Num Tags: {len(tags)}")
    
    # Train your HMM
    a,b,c=construct(sentences,tags,word_dict,tag_dict)

    init = a # TODO: Construct your init matrix
    emission =b # TODO: Construct your emission matrix
    transition =c # TODO: Construct your transition matrix

    # Making sure we have the right shapes
    logging.debug(f"init matrix shape: {init.shape}")
    logging.debug(f"emission matrix shape: {emission.shape}")
    logging.debug(f"transition matrix shape: {transition.shape}")


    # Saving the files for inference
    # We're doing this for you :)
    np.savetxt(args.init, init)
    np.savetxt(args.emission, emission)
    np.savetxt(args.transition, transition)

    return 

# No need to change anything beyond this point
if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    print(args)
    main(args)