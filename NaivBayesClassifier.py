from __future__ import division

from codecs import open
from collections import Counter

import numpy as np


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


def train_nb(documents,labels): #Naive Bayes for training part
    """
    :param documents: training documents list<list<string>>
    :param labels: list<string>
    :return: probability_labels,probability_neg,probability_pos
    """
    labelsCounter=Counter()
    for label in labels:
        labelsCounter[label] +=1

    lableProbability = {}
    for labelKey in labelsCounter:
        lableProbability[labelKey] = labelsCounter[labelKey] / len( labels )

    trainedData = {} # for each label, shows the probability of each word
    for labelKey in labelsCounter:
        labelFreqCounter= Counter()
        for index,document in enumerate(documents):
            if labels[index] == labelKey:
                for token in document:
                    labelFreqCounter[token] += 1
        trainedData[labelKey]=labelFreqCounter


    for key, data in trainedData.items():
        numberOfAllToken= 0
        for token in data:
            numberOfAllToken += data.get(token)
        for token in data:
            data[token] = data.get(token) / numberOfAllToken

    return trainedData, lableProbability


def score_doc_label(document, label, trainedData, lableProbabilty):
    smoothingFactor = 0.5
    words = document.strip().split()
    score = lableProbabilty[label]
    logScore = np.log(lableProbabilty[label])
    for word in words:
        if trainedData.get(label).keys().__contains__(word):
            score += trainedData.get(label).get(word) + smoothingFactor
            logScore += np.log(trainedData.get(label).get(word) + smoothingFactor)

    return score,logScore






