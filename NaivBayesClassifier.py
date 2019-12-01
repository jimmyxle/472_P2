from __future__ import division
from codecs import open
from collections import Counter
import numpy as np


def read_Document(dataFile):
    """
    read dataFile.txt (the original file downloaded from moodle)
    split each line , so we have array of tokens
    :param dataFile: a .txt file
    :return: document,labels
    """
    document = []
    labels = []
    with open(dataFile, encoding='utf-8') as file:
        for line in file:
            words = line.strip().split()
            document.append(words[3:])
            labels.append(words[1])

    return document, labels


def train_nb(documents, labels):  # Naive Bayes for training part
    """
    :param documents: training documents list<list<string>>
    :param labels: list<string>
    :return: probability_labels which is
            p(word|label=pos)=
                (number of times the word is occurred in category pos)/(number of all words in the category)
            and p(word|label=neg) and word stands for all words(tokens) in the review part
    probability_neg that is p(label=pos) and p(label=neg) ,probability_pos
    """
    # count the total number of each label
    labelsCounter = Counter()
    for label in labels:
        labelsCounter[label] += 1
    # calculate the probability of each class (pos,neg)
    lableProbability = {}
    for labelKey in labelsCounter:
        lableProbability[labelKey] = labelsCounter[labelKey] / len(labels)

    # use dictionary (trainedData) to store the probability of each word being observed in  positive or negative review
    trainedData = {}  # for each label, shows the probability of each word
    for labelKey in labelsCounter:  # labelKey is for possible different category we have in an input data
        labelFreqCounter = Counter()
        for index, document in enumerate(documents):
            if labels[index] == labelKey:
                for word in document:
                    labelFreqCounter[word] += 1
        trainedData[labelKey] = labelFreqCounter

    for key, data in trainedData.items():
        numberOfAllToken = 0
        for token in data:
            numberOfAllToken += data.get(token)
        for token in data:
            data[token] = data.get(token) / numberOfAllToken

    return trainedData, lableProbability


def score_doc_label(document, label, trainedData, lableProbabilty):
    """
    Calculate the score of each label with a given document
    :param document: new document or test document
    :param label: test label
    :param trainedData: the probability of each word
    :param lableProbabilty: probability of each label
    :return: logScore :logarithmic value of score
    """
    smoothingFactor = 0.5
    words = {}
    if type(document) is not list:
        words = document.strip().split()
    else:
        words = document
    score = lableProbabilty[label]
    logScore = np.log(lableProbabilty[label])
    for word in words:
        if trainedData.get(label).keys().__contains__(word):
            score *= (trainedData.get(label).get(word) + smoothingFactor)  # score of document/word but not used here for reference
            logScore += np.log(trainedData.get(label).get(word) + smoothingFactor)  # logarithm returns a negative value

    return logScore


# This function claffifies a new document we calculate the score of each category by using the probability of
# observing each word in document and probability of each label The document begongs to the category with higher score
def classify_nb(document, trainedData, lableProbabilty):
    scoresDictionary = {}
    for key in lableProbabilty:
        scoresDictionary[key] = score_doc_label(document, key, trainedData, lableProbabilty)

    bestFitLabel = list(lableProbabilty.keys())[0]
    for label in scoresDictionary.keys():
        if scoresDictionary[label] > scoresDictionary[bestFitLabel]:
            bestFitLabel = label

    return bestFitLabel


# This fuction classifies each document in the test set (being positive/negative review)
# and returns the list of predicted sentiment label , (guessed_labels)
def classify_documents(evaluatingDocs, trainedData, lableProbabilty):
    guessed_labels = []
    for document in evaluatingDocs:
        guessed_labels.append(classify_nb(document, trainedData, lableProbabilty))

    return guessed_labels  # list of predicted sentiment label


# Calculate the accuracy=the number of correctly classified documents/total number of documents
def accuracy(true_labels, guessed_labels):
    correctLabelsCount = 0
    wrongLabelsCount = 0
    for index, value in enumerate(true_labels):
        if true_labels[index] == guessed_labels[index]:
            correctLabelsCount += 1
        else:
            wrongLabelsCount += 1

    return correctLabelsCount / (correctLabelsCount + wrongLabelsCount)
