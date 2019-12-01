import random
import statistics

import NaivBayesClassifier
import sys

num_loops = 50

def run():
    all_documents, all_labels = NaivBayesClassifier.read_Document("dataFile.txt")

    temp = []
    for i in range(len(all_documents)):
        temp.append([all_documents[i], all_labels[i]])
    random.shuffle(temp)
    for i in range(len(all_documents)):
        all_documents[i] = temp[i][0]
        all_labels[i] = temp[i][1]

    split_point = int(0.80 * len(all_documents))  # partition training set set (80%) vs test set (20%)

    training_documents = all_documents[:split_point]  # training set
    training_label = all_labels[:split_point]
    training_label_unique = list(set(training_label))

    evaluation_documents = all_documents[split_point:]  # test set
    evaluation_labels = all_labels[split_point:]

    testDocument = "rien a dire si ce n'est que too $hort et ses potes รงa claque c'est tout"
    testDocument = "love I amazing great super good"
    testDocument = "i loved these movies , and i cant wiat for the third one ! very funny , not suitable for chilren "

    trainedData, lableProbability = NaivBayesClassifier.train_nb(training_documents, training_label)
    # print(trainedData)
    # print(lableProbability)

    # print('logarithmic Score for `pos` label = ',
    #       NaivBayesClassifier.score_doc_label(document=testDocument,
    #                                           label='pos',
    #                                           trainedData=trainedData,
    #                                           lableProbabilty=lableProbability))
    # print('logarithmic Score for `neg` label = ',
    #       NaivBayesClassifier.score_doc_label(testDocument, 'neg', trainedData, lableProbability))

    guessedLabels = NaivBayesClassifier.classify_documents(evaluation_documents, trainedData, lableProbability)
    acc = NaivBayesClassifier.accuracy(evaluation_labels, guessedLabels)
    # print(guessedLabels[0:5])
    print('Accuracy of classifier :', acc)
    # print('Accuracy of classifier :', NaivBayesClassifier.accuracy(training_label, guessedLabels))
    return acc


if __name__ == '__main__':

    accs = []
    for i in range(num_loops):
        accs.append(run())
    max_acc = max(accs)
    min_acc = min(accs)
    mean_acc = statistics.mean(accs)
    med_acc = statistics.median(accs)
    print('---------')
    print('stats: ')
    print('max: ', max_acc)
    print('min: ', min_acc)
    print('mean: ', mean_acc)
    print('median: ', med_acc)
    print('---------')
