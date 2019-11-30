import random

import NaivBayesClassifier
import sys

if __name__ == '__main__':
    all_documents, all_labels = NaivBayesClassifier.read_Document("dataFile.txt")

    temp = []
    for i in range(len(all_documents)):
        temp.append([all_documents[i],all_labels[i]])
    random.shuffle(temp)
    for i in range(len(all_documents)):
        all_documents[i] = temp[i][0]
        all_labels[i]=temp[i][1]

    split_point = int(0.80 * len(all_documents))        # partition training set set (80%) vs test set (20%)

    training_documents = all_documents[:split_point]    # training set
    training_label = all_labels[:split_point]
    training_label_unique = list(set(training_label))

    evaluation_documents = all_documents[split_point:]  # test set
    evaluation_labels = all_labels[split_point:]

    testDocument = "rien a dire si ce n'est que too $hort et ses potes รงa claque c'est tout"
    testDocument = "love I amazing great super good"
    testDocument = "i loved these movies , and i cant wiat for the third one ! very funny , not suitable for chilren "

    trainedData, lableProbability = NaivBayesClassifier.train_nb(training_documents, training_label)
    # print(trainedData)
    # print(str(trainedData), file=open("pOfWordInLabel.txt", "a" ))

    # print(lableProbability)

    print('logarithmic Score for `pos` label = ',
          NaivBayesClassifier.score_doc_label(testDocument, 'pos', trainedData, lableProbability))
    print('logarithmic Score for `neg` label = ',
          NaivBayesClassifier.score_doc_label(testDocument, 'neg', trainedData, lableProbability))

    guessedLabels = NaivBayesClassifier.classify_documents(evaluation_documents, trainedData, lableProbability)

    print(guessedLabels[0:5])
    print('Accuracy of classifier :', NaivBayesClassifier.accuracy(evaluation_labels, guessedLabels))
    # print('Accuracy of classifier :', NaivBayesClassifier.accuracy(training_label, guessedLabels))
