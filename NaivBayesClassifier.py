from __future__ import division

from codecs import open
from collections import Counter

def read_Document(dataFile):
    """
    read dataFile.txt (the original file downloaded from moodle)
    split each line , so we have array of tokens
    :param dataFile:
    :return: document,labels
    """
    document=[]
    labels=[]
    with open (dataFile,encoding='utf-8') as file:
        for line in file:
            words=line.strip().split()
            document.append(words[3:])
            labels.append(words[1])

    return document,labels

def processed_file():
    """
    :return: array of tokens and labels writen to .txt file
    """
    f= open("b.txt", "w")
    dataFile="dataFile.txt"
    processed_Doc=read_Document(dataFile)
    f.write(str(processed_Doc ))
    return f



def train_nb(documents,labels): #Naive Bayes for training part
    """
    :param documents: training documents
    :param labels: training labels
    :return: probability_labels,probability_neg,probability_pos
    """
    labelCounter=Counter()
    for label in labels:
        labelCounter[label] +=1

    TokensOfNeg_label_counter=Counter()
    for l,document in enumerate(documents):
        if labels[l] == 'neg':
            for tokens in document:
                TokensOfNeg_label_counter[tokens] +=1

    TokensOfPos_label_counter=Counter()
    for l,document in enumerate(documents):
        if labels[l]=='pos':
            for tokens in document:
                TokensOfPos_label_counter[tokens] +=1

    print("total number of labels: ",labelCounter)
    #print("number Of Pos Labels: ",TokensOfNeg_label_counter)
    #print("number Of Neg Labels: ",TokensOfPos_label_counter)

    probability_labels={}
    for i in labels:
        probability_labels[i]=labelCounter[i]/(sum([labelCounter[i] for i in labels]))
        print(probability_labels[i])

    probability_neg = {}
    for i in TokensOfNeg_label_counter:
        probability_neg[i]=TokensOfNeg_label_counter[i]/sum(TokensOfNeg_label_counter.values())

    probability_pos={}
    for i in TokensOfPos_label_counter:
        probability_pos[i]=TokensOfPos_label_counter[i]/sum(TokensOfPos_label_counter.values())


    print("Estimating parameters for naive bayes")

    return probability_labels,probability_neg,probability_pos


def score_doc_label(document,label,probability_labels,probability_neg,probability_pos):

    return None