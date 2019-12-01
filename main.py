import NaivBayesClassifier
import sys
if __name__ == '__main__':
    all_documents, all_labels = NaivBayesClassifier.read_Document("dataFile.txt")
    split_point = int(0.80 * len(all_documents))
    training_documents = all_documents[:split_point]
    training_label = all_labels[:split_point]
    training_label_unique = list(set(training_label))
    evaluation_documents = all_documents[split_point:]
    evaluation_labels = all_labels[split_point:]


    trainedData, label_probability = NaivBayesClassifier.train_nb(training_documents, training_label)

    if len(sys.argv) > 2 and sys.argv[1] == "-d":
        print("Testing file ", sys.argv[2])
        test_doc, test_label = NaivBayesClassifier.read_Document(sys.argv[2])
        guessedLabels = NaivBayesClassifier.classify_documents(test_doc, trainedData, label_probability)
        print("Accuracy = ", NaivBayesClassifier.accuracy(test_label, guessedLabels))
    elif sys.argv[1] == "-m":
        print("Testing message = ", sys.argv[2])
        print("positive log score = ", NaivBayesClassifier.score_doc_label(sys.argv[2], 'pos', trainedData, label_probability))
        print("negative log score = ", NaivBayesClassifier.score_doc_label(sys.argv[2], 'neg', trainedData, label_probability))
        print("This document is probably = ", NaivBayesClassifier.classify_nb(sys.argv[2], trainedData, label_probability))

    print('Accuracy of detecting a positive class:',
          NaivBayesClassifier.accuracyOfClassDetection(evaluation_labels, guessedLabels, 'pos'))
    print('Accuracy of detecting a negative class:',
          NaivBayesClassifier.accuracyOfClassDetection(evaluation_labels, guessedLabels, 'neg'))

    print('Accuracy of guessed positive classes:',
          NaivBayesClassifier.accuracyOfGuessedClass(evaluation_labels, guessedLabels, 'pos'))
    print('Accuracy of guessed negative classes:',
          NaivBayesClassifier.accuracyOfGuessedClass(evaluation_labels, guessedLabels, 'neg'))




