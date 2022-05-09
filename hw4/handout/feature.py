import csv
import numpy as np
import sys
import copy
import time

VECTOR_LEN = 300   # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


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
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_dictionary(file):
    """
    Creates a python dict from the model 1 dictionary.

    Parameters:
        file (str): File path to the dictionary for model 1.

    Returns:
        A dictionary indexed by strings, returning an integer index.
    """
    dict_map = np.loadtxt(file, comments=None, encoding='utf-8',
                          dtype=f'U{MAX_WORD_LEN},l')
    return {word: index for word, index in dict_map}


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file for model 2.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map

def update(input_data,output_data,d):
    length=len(input_data)
    length1=len(d)
    #features=[]
    labels=[]
    file=open(output_data, "w")
    for item in input_data:
        label=item[0]
        line=str(label)
        labels.append(label)
        words=item[1].split(' ')
        feature=np.zeros(length1,dtype=int)
        for word in words:
            if word in d:
                feature[d[word]]=1
        for value in feature:
            item="\t{}".format(value)
            line+=item
        line+="\n"
        file.write(line)
    file.close()

def update1(input_data,output_data,word2vec):
    file=open(output_data, "w")
    length=len(input_data)
    vector_length=len(word2vec)
    for i in range(length):
        item=input_data[i]
        label=item[0]
        line=str(label)
        words=item[1].split(' ')
        count=0
        res=np.zeros(300,dtype=int)
        for word in words:
            if word in word2vec:
                res=np.sum([res,word2vec[word]],axis=0)
                count+=1
        k=res/(count)
        for value in k:
            item="\t{}".format(value)
            line+=item
        line+="\n"
        file.write(line)
    file.close()

def main():
    start = time.time() 
    train_input=str(sys.argv[1])
    validation_input=str(sys.argv[2])
    test_input=str(sys.argv[3])

    train_input=load_tsv_dataset(train_input)
    validation_input=load_tsv_dataset(validation_input)
    test_input=load_tsv_dataset(test_input)

    formatted_train_out=str(sys.argv[6])
    formatted_validation_out=str(sys.argv[7])
    formatted_test_out=str(sys.argv[8])

    feature_flag=int(sys.argv[9])
    if feature_flag==1:
        dict_input=str(sys.argv[4])
        d=load_dictionary(dict_input)
        update(train_input,formatted_train_out,d)
        update(validation_input,formatted_validation_out,d)
        update(test_input,formatted_test_out,d)
    if feature_flag==2:
        feature_dictionary_input=str(sys.argv[5])
        feature_dictionary=load_feature_dictionary(feature_dictionary_input)
        update1(train_input,formatted_train_out,feature_dictionary)
        update1(validation_input,formatted_validation_out,feature_dictionary)
        update1(test_input,formatted_test_out,feature_dictionary)
    end = time.time() 
    print(end - start)
if __name__ == "__main__":
	main()