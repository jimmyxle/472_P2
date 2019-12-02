import random
import statistics
import NaivBayesClassifier
import random
import sys

def scramble_data_set():
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
    evaluation_documents = all_documents[split_point:]  # test set
    evaluation_labels = all_labels[split_point:]

    trainedData, lableProbability = NaivBayesClassifier.train_nb(training_documents, training_label)
    guessedLabels = NaivBayesClassifier.classify_documents(evaluation_documents, trainedData, lableProbability)
    acc = NaivBayesClassifier.accuracy(evaluation_labels, guessedLabels)
    #print('Accuracy of classifier :', acc)
    return acc


if __name__ == '__main__':
    all_documents, all_labels = NaivBayesClassifier.read_Document("dataFile.txt")
    split_point = int(0.80 * len(all_documents))  # partition training set set (80%) vs test set (20%)
    training_documents = all_documents[:split_point]  # training set
    training_label = all_labels[:split_point]
    training_label_unique = list(set(training_label))
    evaluation_documents = all_documents[split_point:]  # test set
    evaluation_labels = all_labels[split_point:]
    trainedData, label_probability = NaivBayesClassifier.train_nb(training_documents, training_label)
    if len(sys.argv) == 2 and sys.argv[1] == "-default":
        print("Testing default ")
        guessedLabels = NaivBayesClassifier.classify_documents(evaluation_documents, trainedData, label_probability)
        print("Accuracy = ", NaivBayesClassifier.accuracy(evaluation_labels, guessedLabels))
    elif len(sys.argv) > 2 and sys.argv[1] == "-d":
        print("Testing file ", sys.argv[2])
        test_doc, test_label = NaivBayesClassifier.read_Document(sys.argv[2])
        guessedLabels = NaivBayesClassifier.classify_documents(test_doc, trainedData, label_probability)
        print("Accuracy = ", NaivBayesClassifier.accuracy(test_label, guessedLabels))
    elif len(sys.argv) > 2 and sys.argv[1] == "-m":
        print("Testing message = ", sys.argv[2])
        print("positive log score = ", NaivBayesClassifier.score_doc_label(sys.argv[2], 'pos', trainedData, label_probability))
        print("negative log score = ", NaivBayesClassifier.score_doc_label(sys.argv[2], 'neg', trainedData, label_probability))
        print("This document is probably = ", NaivBayesClassifier.classify_nb(sys.argv[2], trainedData, label_probability))
    elif len(sys.argv) > 2 and sys.argv[1] == "-a":
        guessedLabels = NaivBayesClassifier.classify_documents(evaluation_documents, trainedData, label_probability)
        print('Accuracy of detecting a positive class:', NaivBayesClassifier.accuracyOfClassDetection(evaluation_labels, guessedLabels, 'pos'))
        print('Accuracy of detecting a negative class:', NaivBayesClassifier.accuracyOfClassDetection(evaluation_labels, guessedLabels, 'neg'))

        prec_pos, rec_pos = NaivBayesClassifier.prec_rec(evaluation_labels, guessedLabels, 'pos')
        prec_neg, rec_neg = NaivBayesClassifier.prec_rec(evaluation_labels, guessedLabels, 'neg')

        print('----')
        print('Precision [positive]: ', prec_pos, ' - [negative]: ', prec_neg)
        print('Recall [positive]: ', rec_pos, ' - [negative]: ', rec_neg)
        print('Precision total: ', prec_pos + prec_neg)
        print('Recall total: ', rec_pos + rec_neg)
        print('----')

        num_loops = 5
        accs = []
        for i in range(num_loops):
            accs.append(scramble_data_set())
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
    else:
        print("Naive Baye's Classifier")
        print("Run as : main.py [flag] [arg]")
        print("List of flags:\n'-d'\t Document\n'-m'\t String\n'-a'\t Accuracy statistics. Does not need an argument\n")
        print("Example: \nmain.py -a\nmain.py -d \"ourtest.txt\"\nmain.py -m \"This is a test string.\"")




